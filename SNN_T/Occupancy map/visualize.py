#!/usr/bin/env python3
"""4D occupancy visualizer with on-screen 3D preview and MP4 export.

Renders frames in-memory (no intermediate PNGs written), shows an interactive
window titled "3dopen" while rendering, and writes a single MP4 into
`SNN_T/result/occupancy_viz.mp4` — removing other files in that folder so only
the MP4 remains.
"""
import os
import sys
import argparse
import shutil
import numpy as np

import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from occupancy_processed import Occupancy4DMapProcessor


def render_and_save_mp4(tensor_path=None, out_path=None, fps=2, threshold=0.5, grid_res=0.2, max_T=4, show=True):
    if tensor_path is None:
        candidate_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_maps'))
        files = sorted([os.path.join(candidate_dir, f) for f in os.listdir(candidate_dir) if f.endswith('.npy')])
        if not files:
            raise FileNotFoundError('No tensor files found in output_maps')
        tensor_path = files[-1]

    processor = Occupancy4DMapProcessor(threshold=threshold, grid_resolution=grid_res)
    occupancy, prob = processor.convert_tensor_to_occupancy_map(tensor_path)

    T_avail = occupancy.shape[0]
    T = min(max_T, T_avail)

    # compute global bounds for consistent axes
    all_points = []
    for t in range(T):
        grid = occupancy[t]
        if grid.ndim == 3:
            inds = np.argwhere(grid > 0)
            if inds.size:
                xs = inds[:, 2] * grid_res
                ys = inds[:, 1] * grid_res
                zs = inds[:, 0] * grid_res
                pts = np.stack([xs, ys, zs], axis=1)
                all_points.append(pts)
        else:
            inds = np.argwhere(grid > 0)
            if inds.size:
                xs = inds[:, 1] * grid_res
                ys = inds[:, 0] * grid_res
                zs = np.zeros_like(xs)
                pts = np.stack([xs, ys, zs], axis=1)
                all_points.append(pts)

    if all_points:
        all_pts = np.vstack(all_points)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
    else:
        mins = np.array([-1.0, -1.0, -1.0])
        maxs = np.array([1.0, 1.0, 1.0])

    # prepare interactive figure (reuse for each frame)
    fig = plt.figure(figsize=(8, 6))
    manager = plt.get_current_fig_manager()
    try:
        manager.set_window_title('3dopen')
    except Exception:
        pass
    ax = fig.add_subplot(111, projection='3d')

    frames = []
    for t in range(T):
        grid = occupancy[t]
        if grid.ndim == 3:
            inds = np.argwhere(grid > 0)
            if inds.size:
                xs = inds[:, 2] * grid_res
                ys = inds[:, 1] * grid_res
                zs = inds[:, 0] * grid_res
                pts = np.stack([xs, ys, zs], axis=1)
            else:
                pts = np.zeros((0, 3))
        else:
            inds = np.argwhere(grid > 0)
            if inds.size:
                xs = inds[:, 1] * grid_res
                ys = inds[:, 0] * grid_res
                zs = np.zeros_like(xs)
                pts = np.stack([xs, ys, zs], axis=1)
            else:
                pts = np.zeros((0, 3))

        ax.cla()
        if pts.size:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=1)

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        centroid = pts.mean(axis=0) if pts.size else np.array([0.0, 0.0, 0.0])
        info = f'T={t}  points={len(pts)}\ncentroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})'
        # set per-axes title (more robust across backends and redraws)
        ax.set_title(info)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # Read canvas buffer robustly across backends
        img = None
        try:
            s = fig.canvas.tostring_rgb()
            arr = np.frombuffer(s, dtype=np.uint8)
            img = arr.reshape((h, w, 3))
        except Exception:
            try:
                s = fig.canvas.tostring_argb()
                arr = np.frombuffer(s, dtype=np.uint8).reshape((h, w, 4))
                # ARGB -> RGB
                img = arr[:, :, 1:4]
            except Exception:
                try:
                    buf = fig.canvas.buffer_rgba()
                    arr = np.asarray(buf)
                    img = arr[:, :, :3]
                except Exception:
                    raise

        # save per-frame PNG to results directory
        try:
            results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
            os.makedirs(results_dir, exist_ok=True)
            png_path = os.path.join(results_dir, f'frame_{t:04d}.png')
            imageio.imsave(png_path, img)
        except Exception:
            pass

        frames.append(img)

        if show:
            plt.pause(0.001)

    # ensure results dir inside SNN_T (parent of this file)
    result_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(result_dir, exist_ok=True)
    if out_path is None:
        out_path = os.path.join(result_dir, 'occupancy_viz.mp4')

    # remove any existing video files in result dir (keep per-frame PNGs)
    for f in os.listdir(result_dir):
        fp = os.path.join(result_dir, f)
        try:
            if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                os.remove(fp)
        except Exception:
            pass

    imageio.mimsave(out_path, frames, fps=fps)

    print('Wrote', out_path)
    if show:
        plt.show()
    return out_path


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--fps', type=int, default=2)
    p.add_argument('--T', type=int, default=4)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--grid-res', type=float, default=0.2)
    p.add_argument('--no-show', dest='no_show', action='store_true', help='do not open the interactive 3D window')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    render_and_save_mp4(tensor_path=args.tensor, out_path=args.out, fps=args.fps, threshold=args.threshold, grid_res=args.grid_res, max_T=args.T, show=not args.no_show)
