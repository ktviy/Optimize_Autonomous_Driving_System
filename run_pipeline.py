#!/usr/bin/env python3
"""End-to-end FSD pipeline: KITTI stereo -> depth -> 2.5D BEV map -> occupancy -> motion prediction.

KITTI object (single frames; evaluates depth vs LiDAR and occupancy vs labels):
    python run_pipeline.py object --kitti-root ../Dataset/KITTI --split Pseudo_Lidar/split/subval.txt --out runs/object_subval

KITTI tracking (sequences; adds real ego speed from OXTS and ADE/FDE vs the real future path):
    python run_pipeline.py tracking --tracking-root ../Dataset/KITTI_tracking --seq 0001 --out runs/tracking_0001

Useful flags: --stages depth,bev,motion,eval (default; add snn to also export SNN features)
              --max-frames N (quick test) --device cuda|cpu --no-plot --force (redo finished stages)
Only checks, no work:  python run_pipeline.py object --kitti-root ... --check
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PL = os.path.join(ROOT, 'Pseudo_Lidar')
for sub in ('pipeline', os.path.join('Pseudo_Lidar', 'src', 'compression'), 'Motion_Planning'):
    sys.path.insert(0, os.path.join(ROOT, sub))

DEFAULT_CKPT = os.path.join(PL, 'results', 'sdn_kitti_train_set', 'checkpoint.pth')
ALL_STAGES = ('depth', 'bev', 'snn', 'motion', 'eval')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- preflight
def preflight(args, training_dir, ids, stages):
    problems, notes = [], []
    need = ['numpy', 'scipy', 'torch', 'matplotlib', 'PIL']
    if 'depth' in stages:
        need += ['torchvision', 'configargparse', 'tensorboardX', 'tqdm']
    if 'snn' in stages:
        need += ['spikingjelly', 'timm']
    for mod in need:
        if importlib.util.find_spec(mod) is None:
            problems.append(f"python package '{mod}' missing  ->  pip install -r requirements.txt")
    if 'depth' in stages:
        if not os.path.isfile(args.checkpoint):
            problems.append(f'SDN checkpoint not found: {args.checkpoint}')
        try:
            import torch
            if torch.cuda.is_available():
                notes.append(f'GPU: {torch.cuda.get_device_name(0)}')
            elif args.device != 'cpu':
                notes.append('No CUDA GPU visible: depth will run on CPU (~10 s/frame). '
                             'Check `nvidia-smi` and that torch was installed with CUDA.')
        except Exception as e:  # pragma: no cover
            problems.append(f'torch import failed: {e}')
    for sub in ('image_2', 'image_3', 'calib'):
        d = os.path.join(training_dir, sub)
        if not os.path.isdir(d):
            problems.append(f'missing folder {d}')
            continue
        ext = '.txt' if sub == 'calib' else '.png'
        missing = [i for i in ids if not os.path.isfile(os.path.join(d, i + ext))]
        if missing:
            problems.append(f'{len(missing)}/{len(ids)} frames missing in {d} (first: {missing[0]}{ext})')
    if 'eval' in stages:
        for sub, what in (('label_2', 'occupancy recall'), ('velodyne', 'depth error vs LiDAR')):
            if not os.path.isdir(os.path.join(training_dir, sub)):
                notes.append(f'no {sub}/ -> {what} will be skipped')
    free_gb = shutil.disk_usage(os.path.abspath(args.out) if os.path.exists(args.out) else ROOT).free / 1e9
    est_gb = len(ids) * 1.9e-3 + len(ids) * 0.5e-3  # depth .npy ~1.9 MB + 2.5D map ~0.4 MB per frame
    if free_gb < est_gb * 1.5:
        problems.append(f'only {free_gb:.1f} GB free, need ~{est_gb:.1f} GB')
    notes.append(f'{len(ids)} frames, ~{est_gb:.2f} GB of outputs, {free_gb:.0f} GB free')
    return problems, notes


# ----------------------------------------------------------------------------- stages
def stage_depth(args, training_dir, split_file, depth_dir, ids):
    have = {os.path.splitext(f)[0] for f in os.listdir(depth_dir)} if os.path.isdir(depth_dir) else set()
    todo = list(ids) if args.force else [i for i in ids if i not in have]
    if not todo:
        log(f'depth: all {len(ids)} maps already in {depth_dir} (use --force to redo)')
        return
    if len(todo) < len(ids):  # resume an interrupted run: only the missing frames
        log(f'depth: {len(ids) - len(todo)} maps already done, computing the other {len(todo)}')
        split_file = os.path.join(args.out, 'depth_todo.txt')
        with open(split_file, 'w') as fh:
            fh.write('\n'.join(todo) + '\n')
    cmd = [sys.executable, os.path.join('src', 'main.py'),
           '-c', os.path.join('src', 'configs', 'sdn_kitti_train.config'),
           '--generate_depth_map', '--data_list', os.path.abspath(split_file), '--data_tag', 'pipeline',
           '--datapath', os.path.abspath(training_dir) + os.sep,
           '--resume', os.path.abspath(args.checkpoint), '--pretrain', '',
           '--depth_out_dir', os.path.abspath(depth_dir),
           '--save_path', os.path.abspath(os.path.join(args.out, 'sdn_log')),
           '--device', args.device, '--workers', str(args.workers), '--bval', str(args.batch)]
    if args.dynamic_bs:
        cmd.append('--dynamic_bs')
    log('depth: ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=PL, check=True)


def stage_bev(args, training_dir, depth_dir, bev_dir, ids):
    import torch
    from map_visual import convert_folder
    have = {f.split('_')[0] for f in os.listdir(bev_dir)} if os.path.isdir(bev_dir) else set()
    todo = [i for i in ids if args.force or i not in have]
    if not todo:
        log(f'bev: all {len(ids)} maps already in {bev_dir}')
        return
    dev = torch.device('cuda' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    done, skipped = convert_folder(depth_dir, os.path.join(training_dir, 'calib'), bev_dir,
                                   log_path=os.path.join(args.out, 'ram_log.txt'), device=dev, ids=todo, verbose=False)
    log(f'bev: {done} maps written to {bev_dir}, {skipped} skipped')


def stage_snn(args, bev_dir):
    out = os.path.join(args.out, 'snn_features')
    cmd = [sys.executable, os.path.join(ROOT, 'SNN_T', 'Occupancy map', 'infer_map.py'),
           '--data-dir', bev_dir, '--out-dir', out, '--max-frames', '0']
    log('snn: ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=os.path.join(ROOT, 'SNN_T'), check=True)


def stage_motion(args, bev_dir, motion_dir, ids, car_speeds):
    from run_motion_prediction import predict_files
    files = [os.path.join(bev_dir, f'{i}_25d.npy') for i in ids if os.path.isfile(os.path.join(bev_dir, f'{i}_25d.npy'))]
    log(f'motion: {len(files)} frames, agents {args.agents}')
    t0 = time.time()
    summary = predict_files(files, motion_dir, agents=args.agents, v0_car=args.v0_car, v0_robot=args.v0_robot,
                            car_speeds=car_speeds, plot=not args.no_plot, verbose=args.verbose)
    log(f'motion: done in {time.time() - t0:.0f} s')
    return summary


def stage_eval(args, mode, training_dir, depth_dir, bev_dir, motion_dir, ids, labels, poses, velo_paths):
    import evaluate as ev
    from bev_occupancy import OBSTACLE, load_25d
    from agents import CarAgent
    from planner import MotionPredictor

    report = {}
    # depth vs LiDAR
    drows = []
    for i in ids:
        vp, dp = velo_paths.get(i), os.path.join(depth_dir, i + '.npy')
        if vp and os.path.isfile(vp) and os.path.isfile(dp):
            from kitti_io import read_calib
            pred = np.load(dp)
            gt = ev.lidar_depth_image(vp, read_calib(os.path.join(training_dir, 'calib', i + '.txt')), pred.shape)
            e = ev.depth_errors(pred, gt)
            if e:
                drows.append(e)
    if drows:
        report['depth_vs_lidar'] = dict(frames=len(drows), **ev.mean_dict(drows))

    # occupancy vs labelled objects (agent = car, the same grid the car planner uses)
    if labels:
        pred = MotionPredictor(CarAgent())
        orows = []
        for i in ids:
            bp = os.path.join(bev_dir, f'{i}_25d.npy')
            if i in labels and os.path.isfile(bp):
                grid, _ = pred.build_maps(load_25d(bp))
                orows += ev.occupancy_recall(labels[i], grid, OBSTACLE)
        report['occupancy_recall'] = ev.summarize_recall(orows)

    # motion vs OXTS future path
    if poses is not None:
        from kitti_io import future_ego_path
        n = int(round(CarAgent().horizon / 0.1))  # KITTI tracking/raw run at 10 Hz
        mrows = []
        for t, i in enumerate(ids):
            mp = os.path.join(motion_dir, f'{i}_motion.json')
            gt = future_ego_path(poses, t, n)
            if gt is None or not os.path.isfile(mp):
                continue
            with open(mp) as fh:
                rec = json.load(fh)
            e = ev.motion_errors(rec, gt)
            if e:
                mrows.append(dict(frame=i, **e))
        if mrows:
            report['motion_vs_oxts'] = dict(frames=len(mrows), horizon_s=n * 0.1, **ev.mean_dict(mrows))
            with open(os.path.join(args.out, 'motion_errors_per_frame.json'), 'w') as fh:
                json.dump(mrows, fh, indent=1)

    # planner outcome stats
    recs = []
    for i in ids:
        mp = os.path.join(motion_dir, f'{i}_motion.json')
        if os.path.isfile(mp):
            with open(mp) as fh:
                recs.append(json.load(fh))
    for agent in args.agents:
        rr = [r['agents'][agent] for r in recs if agent in r['agents']]
        if rr:
            report[f'planner_{agent}'] = dict(
                frames=len(rr),
                emergency_stops=int(sum(r['emergency_stop'] for r in rr)),
                planned_stops=int(sum((not r['emergency_stop']) and r['modes'][0]['trajectory'][-1]['speed'] < 0.5 for r in rr)),
                mean_runtime_ms=float(np.mean([r['runtime_ms'] for r in rr])),
                mean_obstacle_frac=float(np.mean([r['occupancy']['obstacle_frac'] for r in rr])),
                mean_camera_height_est_m=float(np.mean([r['occupancy']['camera_height_est'] for r in rr])))
    return report


def write_summary(args, report, mode, n_frames):
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(dict(mode=mode, frames=n_frames, args=vars(args), results=report), fh, indent=1, default=str)
    lines = [f'# FSD pipeline run ({mode}, {n_frames} frames)', '']
    for section, vals in report.items():
        lines.append(f'## {section}')
        for k, v in vals.items():
            lines.append(f'- {k}: {v:.4g}' if isinstance(v, float) else f'- {k}: {v}')
        lines.append('')
    with open(os.path.join(args.out, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(lines))
    print('\n'.join(lines))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    po = sub.add_parser('object', help='KITTI object dataset (single frames)')
    po.add_argument('--kitti-root', required=True, help='folder containing training/ (or the training folder itself)')
    po.add_argument('--split', default=os.path.join(PL, 'split', 'subval.txt'), help='frame id list')
    pt = sub.add_parser('tracking', help='KITTI tracking dataset (sequences + OXTS)')
    pt.add_argument('--tracking-root', required=True, help='folder containing training/ (or training itself)')
    pt.add_argument('--seq', required=True, help='sequence id, e.g. 0001')
    for p in (po, pt):
        p.add_argument('--out', required=True, help='output folder for this run')
        p.add_argument('--stages', default='depth,bev,motion,eval', help=f'comma list from {ALL_STAGES}')
        p.add_argument('--max-frames', type=int, default=0, help='0 = all frames')
        p.add_argument('--checkpoint', default=DEFAULT_CKPT, help='SDN checkpoint (trained on KITTI object train split)')
        p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
        p.add_argument('--workers', type=int, default=4)
        p.add_argument('--batch', type=int, default=4, help='SDN batch size (lower if GPU runs out of memory)')
        p.add_argument('--dynamic_bs', action='store_true', help='stereo baseline from calib instead of 0.54 m')
        p.add_argument('--agents', nargs='+', default=['car', 'robot'])
        p.add_argument('--v0-car', type=float, default=8.0, help='car speed when no OXTS speed is known')
        p.add_argument('--v0-robot', type=float, default=1.0)
        p.add_argument('--no-plot', action='store_true')
        p.add_argument('--force', action='store_true', help='recompute stages whose outputs exist')
        p.add_argument('--verbose', action='store_true')
        p.add_argument('--check', action='store_true', help='only run the preflight checks')
    args = ap.parse_args()
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    bad = [s for s in stages if s not in ALL_STAGES]
    if bad:
        sys.exit(f'unknown stage(s) {bad}; choose from {ALL_STAGES}')
    os.makedirs(args.out, exist_ok=True)

    from kitti_io import (read_object_labels, read_oxts, oxts_to_poses, read_tracking_labels,
                          stage_tracking_sequence)
    labels, poses, car_speeds, velo_paths = {}, None, None, {}

    if args.mode == 'object':
        root = args.kitti_root
        training_dir = os.path.join(root, 'training') if os.path.isdir(os.path.join(root, 'training')) else root
        with open(args.split) as fh:
            ids = [l.strip() for l in fh if l.strip()]
        if args.max_frames:
            ids = ids[:args.max_frames]
        split_file = os.path.join(args.out, 'frames.txt')
        with open(split_file, 'w') as fh:
            fh.write('\n'.join(ids) + '\n')
        lab_dir = os.path.join(training_dir, 'label_2')
        if os.path.isdir(lab_dir):
            labels = {i: read_object_labels(os.path.join(lab_dir, i + '.txt')) for i in ids
                      if os.path.isfile(os.path.join(lab_dir, i + '.txt'))}
        velo_paths = {i: os.path.join(training_dir, 'velodyne', i + '.bin') for i in ids}
    else:
        root = args.tracking_root
        trk = os.path.join(root, 'training') if os.path.isdir(os.path.join(root, 'training')) else root
        training_dir = os.path.join(args.out, 'staged')
        ids = stage_tracking_sequence(trk, args.seq, training_dir, args.max_frames or None)
        split_file = os.path.join(args.out, 'frames.txt')
        with open(split_file, 'w') as fh:
            fh.write('\n'.join(ids) + '\n')
        oxts_path = os.path.join(trk, 'oxts', f'{args.seq}.txt')
        if os.path.isfile(oxts_path):
            ox = read_oxts(oxts_path)[:len(ids)]
            poses = oxts_to_poses(ox)
            car_speeds = {i: float(max(ox[k, 8], 0.0)) for k, i in enumerate(ids)}  # vf = forward speed
            with open(os.path.join(args.out, 'oxts_speeds.json'), 'w') as fh:
                json.dump(car_speeds, fh, indent=1)
        else:
            log(f'WARNING: {oxts_path} missing: no real speed, no ADE/FDE')
        lab_path = os.path.join(trk, 'label_02', f'{args.seq}.txt')
        if os.path.isfile(lab_path):
            per_frame = read_tracking_labels(lab_path)
            labels = {i: per_frame.get(int(i), []) for i in ids}
        velo_paths = {i: os.path.join(trk, 'velodyne', args.seq, i + '.bin') for i in ids}

    problems, notes = preflight(args, training_dir, ids, stages)
    for n in notes:
        log('note: ' + n)
    if problems:
        for p in problems:
            log('PROBLEM: ' + p)
        sys.exit('preflight failed: fix the problems above')
    log('preflight OK')
    if args.check:
        return

    depth_dir = os.path.join(args.out, 'depth')
    bev_dir = os.path.join(args.out, 'bev25d')
    motion_dir = os.path.join(args.out, 'motion')
    if 'depth' in stages:
        stage_depth(args, training_dir, split_file, depth_dir, ids)
    if 'bev' in stages:
        stage_bev(args, training_dir, depth_dir, bev_dir, ids)
    if 'snn' in stages:
        stage_snn(args, bev_dir)
    if 'motion' in stages:
        stage_motion(args, bev_dir, motion_dir, ids, car_speeds)
    if 'eval' in stages:
        report = stage_eval(args, args.mode, training_dir, depth_dir, bev_dir, motion_dir, ids, labels, poses, velo_paths)
        write_summary(args, report, args.mode, len(ids))
    log(f'finished; outputs in {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
