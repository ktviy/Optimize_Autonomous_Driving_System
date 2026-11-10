#!/usr/bin/env python3
"""Render a tiny fake KITTI dataset to smoke-test the whole pipeline without downloading KITTI.

It ray-casts a textured world (flat road, box-shaped "cars", a far wall) into a
calibrated stereo pair, so the real SDN stereo network can estimate depth from it.

Writes the same folder layouts as the real datasets:

  <out>/object/training/{image_2,image_3,calib,label_2,velodyne,depth_gt}/000000.*
  <out>/object/split.txt
  <out>/tracking/training/{image_02,image_03}/0000/000000.png
  <out>/tracking/training/{calib/0000.txt, oxts/0000.txt, label_02/0000.txt}

depth_gt holds the exact rendered depth (for checking SDN), velodyne a fake LiDAR scan.
This is test data only: real results need real KITTI.
"""
import argparse
import os

import numpy as np
from PIL import Image

# KITTI 2011_09_26 calibration (cam2 = left colour, cam3 = right colour)
F, CX, CY = 721.5377, 609.5593, 172.854
P2_TX, P3_TX = 44.85728, -339.5242      # P[0,3]
BASELINE = (P2_TX - P3_TX) / F          # 0.532 m, cam3 sits to the right of cam2
W, H = 1242, 375
CAM_HEIGHT = 1.65
R0 = np.eye(3)
TR_VELO_CAM = np.array([[0, -1, 0, 0], [0, 0, -1, -0.08], [1, 0, 0, -0.27]], float)  # velo x fwd, y left, z up


def hash_noise(ix, iy, seed):
    v = np.sin(ix * 12.9898 + iy * 78.233 + seed * 37.719) * 43758.5453
    return v - np.floor(v)


def texture(a, b, seed):
    """Multi-scale blocky noise on a surface parameterised by (a, b) metres; values 0..1."""
    t = np.zeros_like(a)
    for cell, wgt in ((0.04, 0.45), (0.15, 0.3), (0.6, 0.25)):
        t += wgt * hash_noise(np.floor(a / cell), np.floor(b / cell), seed + cell)
    return t


def ray_cast(boxes, cam_x):
    """Per-pixel depth (camera z) and texture for a camera at (cam_x, 0, 0) looking +z."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    dx, dy, dz = (u - CX) / F, (v - CY) / F, np.ones_like(u)
    t_best = np.full(u.shape, np.inf)
    tex = np.zeros(u.shape)

    # far wall at z = 75 m
    t = 75.0 / dz
    hit = t < t_best
    t_best[hit] = t[hit]
    tex[hit] = texture((cam_x + dx * t)[hit], (dy * t)[hit], 1)

    # road plane y = +CAM_HEIGHT (camera y points down)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(dy > 1e-6, CAM_HEIGHT / dy, np.inf)
    hit = t < t_best
    t_best[hit] = t[hit]
    tex[hit] = 0.6 * texture((cam_x + dx * t)[hit], (dz * t)[hit], 2) + 0.2

    # axis-aligned boxes (slab test)
    for k, (bx, bz, w, l, h) in enumerate(boxes):
        lo = np.array([bx - w / 2 - cam_x, CAM_HEIGHT - h, bz - l / 2])
        hi = np.array([bx + w / 2 - cam_x, CAM_HEIGHT, bz + l / 2])
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = [(lo[i] / d, hi[i] / d) for i, d in enumerate((dx, dy, dz))]
        tmin = np.maximum.reduce([np.minimum(a, b) for a, b in t1])
        tmax = np.minimum.reduce([np.maximum(a, b) for a, b in t1])
        hit = (tmax >= tmin) & (tmin > 0) & (tmin < t_best)
        t_best[hit] = tmin[hit]
        px, py, pz = (cam_x + dx * tmin)[hit], (dy * tmin)[hit], (dz * tmin)[hit]
        tex[hit] = 0.8 * texture(px + pz, py, 10 + k) + 0.1

    depth = t_best * dz  # camera z
    return depth, tex


def render_pair(boxes):
    dl, tl = ray_cast(boxes, 0.0)
    _, tr = ray_cast(boxes, BASELINE)
    to_img = lambda t: Image.fromarray((np.clip(t, 0, 1)[..., None] * np.array([235, 225, 215])).astype(np.uint8))
    return to_img(tl), to_img(tr), dl


def calib_text(tracking=False):
    P0 = f"{F} 0 {CX} 0 0 {F} {CY} 0 0 0 1 0"
    P2 = f"{F} 0 {CX} {P2_TX} 0 {F} {CY} 0 0 0 1 0"
    P3 = f"{F} 0 {CX} {P3_TX} 0 {F} {CY} 0 0 0 1 0"
    r0 = ' '.join(f'{x:.6e}' for x in R0.ravel())
    tv = ' '.join(f'{x:.6e}' for x in TR_VELO_CAM.ravel())
    if tracking:  # tracking calib: "R_rect" / "Tr_velo_cam" keys WITHOUT a colon
        return f"P0: {P0}\nP1: {P0}\nP2: {P2}\nP3: {P3}\nR_rect {r0}\nTr_velo_cam {tv}\nTr_imu_velo {tv}\n"
    return f"P0: {P0}\nP1: {P0}\nP2: {P2}\nP3: {P3}\nR0_rect: {r0}\nTr_velo_to_cam: {tv}\nTr_imu_to_velo: {tv}\n"


def label_lines(boxes, kinds):
    out = []
    for (bx, bz, w, l, h), kind in zip(boxes, kinds):
        # KITTI: type trunc occ alpha x1 y1 x2 y2 h w l x y z ry  (x,y,z = bottom centre, camera coords)
        out.append(f"{kind} 0.00 0 0.00 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {bx:.2f} {CAM_HEIGHT:.2f} {bz:.2f} {np.pi/2:.2f}")
    return "\n".join(out) + "\n"


def fake_velodyne(depth):
    """Sparse LiDAR-like points from the rendered depth (every 4th row), in velodyne coords."""
    v, u = np.mgrid[0:H:4, 0:W:2]
    z = depth[v, u]
    ok = z < 70
    x_c, y_c, z_c = (u[ok] - CX) / F * z[ok], (v[ok] - CY) / F * z[ok], z[ok]
    cam = np.stack([x_c, y_c, z_c], 1)
    # invert Tr_velo_cam: cam = R velo + t -> velo = R^T (cam - t)
    R, t = TR_VELO_CAM[:, :3], TR_VELO_CAM[:, 3]
    velo = (cam - t) @ R
    return np.concatenate([velo, np.ones((len(velo), 1))], 1).astype(np.float32)


OBJECT_SCENES = [
    # (x, z, width, length, height), kinds
    ([(0.3, 14.0, 1.8, 4.2, 1.55), (-4.5, 22.0, 1.8, 4.2, 1.6), (4.2, 9.0, 0.6, 0.6, 1.75)], ['Car', 'Car', 'Pedestrian']),
    ([(-0.5, 25.0, 1.8, 4.5, 1.6), (3.8, 12.0, 1.9, 4.6, 1.7)], ['Car', 'Van']),
]


def write_object(root, n_frames):
    tr = os.path.join(root, 'object', 'training')
    for d in ('image_2', 'image_3', 'calib', 'label_2', 'velodyne', 'depth_gt'):
        os.makedirs(os.path.join(tr, d), exist_ok=True)
    ids = []
    for i in range(n_frames):
        boxes, kinds = OBJECT_SCENES[i % len(OBJECT_SCENES)]
        fid = f"{i:06d}"
        l, r, depth = render_pair(boxes)
        l.save(os.path.join(tr, 'image_2', fid + '.png'))
        r.save(os.path.join(tr, 'image_3', fid + '.png'))
        np.save(os.path.join(tr, 'depth_gt', fid + '.npy'), depth.astype(np.float32))
        fake_velodyne(depth).tofile(os.path.join(tr, 'velodyne', fid + '.bin'))
        open(os.path.join(tr, 'calib', fid + '.txt'), 'w').write(calib_text())
        open(os.path.join(tr, 'label_2', fid + '.txt'), 'w').write(label_lines(boxes, kinds))
        ids.append(fid)
    open(os.path.join(root, 'object', 'split.txt'), 'w').write("\n".join(ids) + "\n")
    return tr


def write_tracking(root, n_frames, speed=6.0, dt=0.1):
    tr = os.path.join(root, 'tracking', 'training')
    seq = '0000'
    for d in ('image_02', 'image_03', 'depth_gt'):
        os.makedirs(os.path.join(tr, d, seq), exist_ok=True)
    os.makedirs(os.path.join(tr, 'calib'), exist_ok=True)
    os.makedirs(os.path.join(tr, 'oxts'), exist_ok=True)
    os.makedirs(os.path.join(tr, 'label_02'), exist_ok=True)
    open(os.path.join(tr, 'calib', seq + '.txt'), 'w').write(calib_text(tracking=True))
    world = [(1.0, 30.0, 1.8, 4.2, 1.55), (-4.0, 18.0, 1.8, 4.2, 1.6), (4.5, 45.0, 1.9, 4.5, 1.7)]
    lat0, lon0 = 49.0, 8.4
    er = 6378137.0
    oxts, labels = [], []
    for i in range(n_frames):
        ego_z = speed * dt * i                       # ego drives straight along world +z (north)
        boxes = [(x, z - ego_z, w, l, h) for x, z, w, l, h in world]
        l, r, depth = render_pair(boxes)
        fid = f"{i:06d}"
        l.save(os.path.join(tr, 'image_02', seq, fid + '.png'))
        r.save(os.path.join(tr, 'image_03', seq, fid + '.png'))
        np.save(os.path.join(tr, 'depth_gt', seq, fid + '.npy'), depth.astype(np.float32))
        lat = lat0 + np.degrees(ego_z / er)
        vals = [lat, lon0, 110.0, 0.0, 0.0, np.pi / 2,   # lat lon alt roll pitch yaw (yaw = heading east->north)
                speed, 0.0, speed, 0.0, 0.0] + [0.0] * 12 + [0.5, 0.05, 4, 10, 4, 4, 4]
        oxts.append(' '.join(f'{v:.10g}' for v in vals))
        for k, (x, z, w, ln, h) in enumerate(boxes):
            labels.append(f"{i} {k} Car 0 0 0.0 0 0 0 0 {h:.2f} {w:.2f} {ln:.2f} {x:.2f} {CAM_HEIGHT:.2f} {z:.2f} {np.pi/2:.2f}")
    open(os.path.join(tr, 'oxts', seq + '.txt'), 'w').write("\n".join(oxts) + "\n")
    open(os.path.join(tr, 'label_02', seq + '.txt'), 'w').write("\n".join(labels) + "\n")
    return tr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--object-frames', type=int, default=2)
    p.add_argument('--tracking-frames', type=int, default=0)
    a = p.parse_args()
    if a.object_frames:
        print('object  ->', write_object(a.out, a.object_frames))
    if a.tracking_frames:
        print('tracking->', write_tracking(a.out, a.tracking_frames))


if __name__ == '__main__':
    main()
