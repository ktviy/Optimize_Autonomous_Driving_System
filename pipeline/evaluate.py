"""Evaluation for each pipeline stage.

  depth     : SDN depth vs LiDAR (velodyne) projected into the image
  occupancy : are labelled KITTI objects marked as obstacles in the occupancy grid?
  motion    : predicted ego trajectory vs the real future path from OXTS (ADE / FDE)
"""
import os

import numpy as np

from kitti_io import OBJECT_CLASSES, velo_to_cam_matrix

RANGE_BINS = ((0, 20), (20, 35), (35, 50))


# ----------------------------------------------------------------------------- depth
def lidar_depth_image(velo_path, calib, shape):
    """Sparse ground-truth depth image from a velodyne .bin (0 where no point)."""
    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    cam = (velo_to_cam_matrix(calib) @ np.c_[pts, np.ones(len(pts))].T)[:3]
    cam = cam[:, cam[2] > 1.0]
    P2 = calib['P2'].reshape(3, 4)
    uvw = P2 @ np.r_[cam, np.ones((1, cam.shape[1]))]
    u, v = np.round(uvw[0] / uvw[2]).astype(int), np.round(uvw[1] / uvw[2]).astype(int)
    H, W = shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = np.zeros(shape, np.float32)
    order = np.argsort(-cam[2, ok])  # nearest point wins
    depth[v[ok][order], u[ok][order]] = cam[2, ok][order]
    return depth


def depth_errors(pred, gt, max_depth=80.0):
    m = (gt > 1.0) & (gt < max_depth)
    if not m.any():
        return None
    p, g = pred[m], gt[m]
    e = np.abs(p - g)
    out = dict(n=int(m.sum()), mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
               abs_rel=float((e / g).mean()), delta1=float((np.maximum(p / g, g / p) < 1.25).mean()))
    for lo, hi in RANGE_BINS:
        mm = (g >= lo) & (g < hi)
        out[f'mae_{lo}_{hi}m'] = float(e[mm].mean()) if mm.any() else None
    return out


# ----------------------------------------------------------------------------- occupancy
def box_footprint_mask(obj, grid, margin=0.3):
    """Boolean [H, W] of BEV cells inside an object's (rotated) footprint, grown by `margin` m."""
    H, W = grid.shape
    rows, cols = np.mgrid[0:H, 0:W]
    x, z = grid.cell_to_world(rows, cols)
    dx, dz = x - obj['x'], z - obj['z']
    c, s = np.cos(obj['ry']), np.sin(obj['ry'])
    # KITTI ry rotates about camera y; box length runs along the object's heading
    along = c * dx - s * dz
    across = s * dx + c * dz
    return (np.abs(along) <= obj['l'] / 2 + margin) & (np.abs(across) <= obj['w'] / 2 + margin)


def occupancy_recall(objects, grid, obstacle_value, min_cells=3):
    """Per evaluable object: was it seen (any observed cell) and marked as an obstacle?"""
    x_lo, x_hi, z_lo, z_hi = grid.extent
    rows = []
    for o in objects:
        if o['type'] not in OBJECT_CLASSES:
            continue
        if not (x_lo + 0.5 < o['x'] < x_hi - 0.5 and z_lo + 1.0 < o['z'] < z_hi - 0.5):
            continue
        if o['occluded'] >= 2 or o['truncated'] > 0.5:  # KITTI "hard" visibility cut
            continue
        m = box_footprint_mask(o, grid)
        n_obst = int((grid.state[m] == obstacle_value).sum())
        rows.append(dict(type=o['type'], range_m=float(np.hypot(o['x'], o['z'])),
                         obstacle_cells=n_obst, detected=n_obst >= min_cells))
    return rows


def summarize_recall(rows):
    if not rows:
        return {}
    out = dict(n_objects=len(rows), recall=float(np.mean([r['detected'] for r in rows])))
    for cls in sorted({r['type'] for r in rows}):
        rr = [r for r in rows if r['type'] == cls]
        out[f'recall_{cls}'] = float(np.mean([r['detected'] for r in rr]))
        out[f'n_{cls}'] = len(rr)
    for lo, hi in RANGE_BINS:
        rr = [r for r in rows if lo <= r['range_m'] < hi]
        if rr:
            out[f'recall_{lo}_{hi}m'] = float(np.mean([r['detected'] for r in rr]))
    return out


# ----------------------------------------------------------------------------- motion
def trajectory_errors(pred_xz, gt_xz):
    """pred_xz, gt_xz: [n, 2] displacements from the start (same time steps)."""
    d = np.linalg.norm(pred_xz - gt_xz, axis=1)
    return float(d.mean()), float(d[-1])


def motion_errors(record, gt_xz, dt=0.1, agent='car'):
    """ADE/FDE of the most likely mode, min over modes, and a constant-velocity baseline."""
    a = record['agents'][agent]
    n = len(gt_xz)
    start_z = a.get('start_z', 0.0)
    res = {}
    modes = []
    for m in a['modes']:
        traj = m['trajectory'][1:n + 1]
        if len(traj) < n:
            return None
        pred = np.array([[p['x'], p['z'] - start_z] for p in traj])
        modes.append(trajectory_errors(pred, gt_xz))
    res['ade'], res['fde'] = modes[0]
    res['min_ade'] = min(m[0] for m in modes)
    res['min_fde'] = min(m[1] for m in modes)
    v0 = a.get('v0', 0.0)
    cv = np.stack([np.zeros(n), v0 * dt * np.arange(1, n + 1)], 1)
    res['cv_ade'], res['cv_fde'] = trajectory_errors(cv, gt_xz)
    res['emergency_stop'] = a['emergency_stop']
    res['gt_distance_m'] = float(np.linalg.norm(gt_xz[-1]))
    return res


def mean_dict(rows):
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
    return out
