"""Readers for the KITTI object and tracking formats used by run_pipeline.py.

Coordinate frames
  camera (rectified cam2): x right, y down, z forward   <- everything in this repo
  IMU / OXTS:              x forward, y left, z up
"""
import os
import shutil

import numpy as np

ER = 6378137.0  # earth radius used by the KITTI devkit (m)
OBJECT_CLASSES = ('Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram')


# ----------------------------------------------------------------------------- calib
def read_calib(path):
    """Parse KITTI calib files of BOTH formats into {key: np.array}.

    object:   'P2: ...', 'R0_rect: ...', 'Tr_velo_to_cam: ...'
    tracking: 'P2: ...', 'R_rect ...',   'Tr_velo_cam ...'   (no colon!)
    raw:      'P_rect_02: ...' (calib_cam_to_cam.txt)
    Keys are normalised to P2, P3, R0_rect, Tr_velo_to_cam.
    """
    out = {}
    alias = {'R_rect': 'R0_rect', 'Tr_velo_cam': 'Tr_velo_to_cam', 'P_rect_02': 'P2', 'P_rect_03': 'P3',
             'R_rect_00': 'R0_rect', 'Tr_imu_velo': 'Tr_imu_to_velo'}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            first = line.split(None, 1)[0]
            if ':' in first:                      # "P2: 7.2e+02 ..." or "P2:7.2e+02"
                key, val = line.split(':', 1)
            else:                                 # "R_rect 9.99e-01 ..." (tracking format)
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, val = parts
            key = alias.get(key.strip(), key.strip())
            try:
                out[key] = np.array([float(x) for x in val.split()])
            except ValueError:
                pass
    if 'P2' not in out:
        raise ValueError(f'{path}: no P2 / P_rect_02 entry')
    return out


def write_object_calib(calib, path):
    """Write a calib dict in the KITTI *object* format (what the SDN loader expects)."""
    lines = []
    for key in ('P0', 'P1', 'P2', 'P3', 'R0_rect', 'Tr_velo_to_cam', 'Tr_imu_to_velo'):
        if key in calib:
            lines.append(f"{key}: " + ' '.join(f'{v:.12e}' for v in calib[key]))
    if 'P3' not in calib:  # SDN only needs P2 (and P3 for --dynamic_bs)
        lines.append("P3: " + ' '.join(f'{v:.12e}' for v in calib['P2']))
    with open(path, 'w') as fh:
        fh.write("\n".join(lines) + "\n")


def velo_to_cam_matrix(calib):
    """4x4 transform velodyne -> rectified camera."""
    T = np.eye(4)
    T[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)
    R = np.eye(4)
    R[:3, :3] = calib['R0_rect'].reshape(3, 3) if 'R0_rect' in calib else np.eye(3)
    return R @ T


# ----------------------------------------------------------------------------- labels
def _label_dict(p):
    # type trunc occ alpha x1 y1 x2 y2 h w l x y z ry [score]
    return dict(type=p[0], truncated=float(p[1]), occluded=int(float(p[2])),
                h=float(p[8]), w=float(p[9]), l=float(p[10]),
                x=float(p[11]), y=float(p[12]), z=float(p[13]), ry=float(p[14]))


def read_object_labels(path):
    objs = []
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 15 and p[0] != 'DontCare':
                objs.append(_label_dict(p))
    return objs


def read_tracking_labels(path):
    """label_02/<seq>.txt -> {frame_index: [objects]} (frame, track_id, then the object format)."""
    frames = {}
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) >= 17 and p[2] != 'DontCare':
                frames.setdefault(int(p[0]), []).append(dict(_label_dict(p[2:]), track_id=int(p[1])))
    return frames


# ----------------------------------------------------------------------------- oxts
def read_oxts(path):
    """oxts/<seq>.txt (tracking) -> array [N, 30]. Field 8 = forward velocity vf (m/s)."""
    return np.loadtxt(path, ndmin=2)


def read_oxts_raw_dir(oxts_dir):
    """raw dataset: oxts/data/0000000000.txt ... -> array [N, 30]."""
    files = sorted(f for f in os.listdir(oxts_dir) if f.endswith('.txt'))
    return np.stack([np.loadtxt(os.path.join(oxts_dir, f)) for f in files])


def oxts_to_poses(oxts):
    """KITTI devkit convertOxtsToPose: list of 4x4 IMU->world poses (Mercator, metres)."""
    lat0 = oxts[0, 0]
    scale = np.cos(np.radians(lat0))
    poses = []
    for row in oxts:
        lat, lon, alt, roll, pitch, yaw = row[:6]
        tx = scale * np.radians(lon) * ER
        ty = scale * ER * np.log(np.tan(np.radians(90 + lat) / 2))
        cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = Rz @ Ry @ Rx
        T[:3, 3] = [tx, ty, alt]
        poses.append(T)
    return poses


def future_ego_path(poses, t, n_steps):
    """Ego positions at frames t+1..t+n relative to frame t, in CAMERA axes (x right, z forward).
    Returns [n, 2] or None if the sequence ends too early."""
    if t + n_steps >= len(poses):
        return None
    R, p0 = poses[t][:3, :3], poses[t][:3, 3]
    rel = np.stack([R.T @ (poses[t + k][:3, 3] - p0) for k in range(1, n_steps + 1)])  # IMU axes
    return np.stack([-rel[:, 1], rel[:, 0]], 1)  # x_cam = -y_imu (left->right), z_cam = x_imu


# ----------------------------------------------------------------------------- tracking -> object layout
def stage_tracking_sequence(tracking_training_dir, seq, staged_dir, frames=None):
    """Make an object-style folder (image_2/, image_3/, calib/<frame>.txt) for one tracking sequence,
    so the unchanged SDN loader and 2.5D converter can read it. Images are hard-linked when possible.
    Returns the list of frame ids (e.g. '000000')."""
    src2 = os.path.join(tracking_training_dir, 'image_02', seq)
    src3 = os.path.join(tracking_training_dir, 'image_03', seq)
    if not os.path.isdir(src3):
        raise FileNotFoundError(f'{src3} missing: the tracking RIGHT images (data_tracking_image_3.zip) are needed for stereo')
    calib = read_calib(os.path.join(tracking_training_dir, 'calib', f'{seq}.txt'))
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(src2) if f.endswith('.png'))
    if frames is not None:
        ids = ids[:frames]
    for sub in ('image_2', 'image_3', 'calib'):
        os.makedirs(os.path.join(staged_dir, sub), exist_ok=True)
    for fid in ids:
        for src, sub in ((src2, 'image_2'), (src3, 'image_3')):
            dst = os.path.join(staged_dir, sub, fid + '.png')
            if not os.path.exists(dst):
                try:
                    os.link(os.path.join(src, fid + '.png'), dst)
                except OSError:
                    shutil.copyfile(os.path.join(src, fid + '.png'), dst)
        write_object_calib(calib, os.path.join(staged_dir, 'calib', fid + '.txt'))
    return ids
