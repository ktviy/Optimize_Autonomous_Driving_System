"""Depth maps (from SDN) -> pseudo-LiDAR points -> 2.5D BEV maps [1, 2, 512, 208].

Usage (from FSD/):
    python Pseudo_Lidar/src/compression/map_visual.py \
        --depth-dir Pseudo_Lidar/results/sdn_kitti_train_set/depth_maps/val \
        --calib-dir ../Dataset/KITTI/training/calib \
        --out-dir   Pseudo_Lidar/results/processed_25d

The grid ranges below are fixed: SNN_T, Transformer and Motion_Planning all assume
x in [-10, 10] m, z in [0, 50] m, 0.1 m cells, y (camera, down) normalized over [-10, 10] m.
"""
import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from projection import depth_to_3d_vector  # noqa: E402
from voxel_to_25d import voxel_to_25d  # noqa: E402
from RAM_measure import generate_benchmark_table  # noqa: E402

X_RANGE = (-10, 10)
Y_RANGE = (-10, 10)
Z_RANGE = (0, 50)
RES = 0.1

HERE = os.path.dirname(os.path.abspath(__file__))
PL_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))


# --- 1. READ CALIB KITTI DATASET ---
def read_kitti_calib(calib_file_path):
    """Returns (focal length, |P2 x-offset| in m (NOT the stereo baseline; unused), cx, cy).
    Works for KITTI object calib ('P2: ...') and tracking/raw calib ('P2: ...' or 'P_rect_02: ...')."""
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('P2:') or line.startswith('P2 ') or line.startswith('P_rect_02:'):
            p2_data = np.array([float(x) for x in line.split(':', 1)[1].split()] if ':' in line
                               else [float(x) for x in line.split()[1:]]).reshape(3, 4)
            f_length = p2_data[0, 0]
            cx_val = p2_data[0, 2]
            cy_val = p2_data[1, 2]
            b_val = np.abs(p2_data[0, 3] / f_length)
            return f_length, b_val, cx_val, cy_val
    raise ValueError(f"Cannot find P2 in calibration file {calib_file_path}")


def depth_file_to_25d(depth_path, calib_path, device):
    f_val, B_val, cx_val, cy_val = read_kitti_calib(calib_path)
    depth_np = np.load(depth_path)  # SDN outputs DEPTH in metres (not disparity)
    depth_tensor = torch.from_numpy(depth_np).to(device).float()
    point_cloud_3d = depth_to_3d_vector(depth_tensor, f_val, B_val, cx_val, cy_val)
    grid_25d = voxel_to_25d(point_cloud_3d, res=RES, x_range=X_RANGE, y_range=Y_RANGE, z_range=Z_RANGE)
    return point_cloud_3d, grid_25d


def convert_folder(depth_dir, calib_dir, out_dir, log_path=None, device=None, ids=None, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    npy_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    if ids is not None:
        ids = set(ids)
        npy_files = [p for p in npy_files if os.path.basename(p).split('.')[0] in ids]
    done, skipped = 0, 0
    for npy_path in npy_files:
        file_id = os.path.basename(npy_path).split('.')[0]
        calib_path = os.path.join(calib_dir, f"{file_id}.txt")
        if not os.path.exists(calib_path):
            print(f"Calibration file {calib_path} does not exist. Skipping {file_id}.")
            skipped += 1
            continue
        point_cloud_3d, grid_25d = depth_file_to_25d(npy_path, calib_path, device)
        np.save(os.path.join(out_dir, f"{file_id}_25d.npy"), grid_25d.cpu().numpy())
        done += 1
        if verbose:
            print(f"Processed {file_id} -> {tuple(grid_25d.shape)}")

        # --- PERFORMANCE BENCHMARKING (RAM of dense point cloud vs 2.5D map) ---
        if log_path:
            with open(log_path, "a") as f:
                f.write(f"\n--- Log Date: {datetime.now():%Y-%m-%d %H:%M:%S} | frame {file_id} ---\n")
                f.write(generate_benchmark_table(point_cloud_3d, grid_25d) + "\n")
                f.write("-" * 30 + "\n")
    return done, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--depth-dir', default=os.path.join(PL_ROOT, 'results', 'sdn_kitti_train_set', 'depth_maps', 'val'))
    p.add_argument('--calib-dir', default=os.path.join(PL_ROOT, '..', '..', 'Dataset', 'KITTI', 'training', 'calib'))
    p.add_argument('--out-dir', default=os.path.join(PL_ROOT, 'results', 'processed_25d'))
    p.add_argument('--log', default=os.path.join(HERE, 'performance_log.txt'), help="'' disables the RAM log")
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())
                          else 'cpu')
    print(f"Run on the device: {device}")
    done, skipped = convert_folder(args.depth_dir, args.calib_dir, args.out_dir, args.log or None, device)
    print(f"Done: {done} maps written to {args.out_dir}, {skipped} skipped (no calib)")


if __name__ == "__main__":
    main()
