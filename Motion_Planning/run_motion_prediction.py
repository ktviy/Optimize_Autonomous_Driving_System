#!/usr/bin/env python3
"""Predict the future motion of an autonomous car and/or robot on KITTI 2.5D maps.

Examples
    python run_motion_prediction.py                       # first 10 frames, car + robot
    python run_motion_prediction.py --frames 000048 000076 --agents car --v0-car 10
    python run_motion_prediction.py --num 50 --no-plot    # batch, JSON only
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from agents import AGENTS  # noqa: E402
from bev_occupancy import load_25d  # noqa: E402
from planner import MotionPredictor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.join(HERE, "..", "Pseudo_Lidar", "results", "processed_25d"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "outputs"))
    p.add_argument("--frames", nargs="*", help="frame ids, e.g. 000048 000076")
    p.add_argument("--num", type=int, default=10, help="number of frames when --frames is not given (0 = all)")
    p.add_argument("--speeds-json", default=None,
                   help='per-frame car speed {"000123": 7.9, ...} (m/s), e.g. from OXTS; overrides --v0-car')
    p.add_argument("--agents", nargs="+", default=["car", "robot"], choices=list(AGENTS))
    p.add_argument("--v0-car", type=float, default=8.0, help="initial car speed (m/s)")
    p.add_argument("--v0-robot", type=float, default=1.0, help="initial robot speed (m/s)")
    p.add_argument("--modes", type=int, default=3, help="number of predicted trajectory modes")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def mode_to_json(m, dt):
    s = m["states"]
    return dict(
        probability=round(m["prob"], 4),
        cost=None if not np.isfinite(m["cost"]) else round(m["cost"], 3),
        terms={k: round(v, 4) for k, v in m["terms"].items()},
        trajectory=[dict(t=round(i * dt, 2), x=round(float(r[0]), 3), z=round(float(r[1]), 3),
                         heading_rad=round(float(r[2]), 4), speed=round(float(r[3]), 3))
                    for i, r in enumerate(s)],
    )


def list_maps(data_dir, frames=None, num=0):
    files = sorted(glob.glob(os.path.join(data_dir, "*_25d.npy")))
    if frames:
        wanted = set(frames)
        files = [f for f in files if os.path.basename(f).split("_")[0] in wanted]
    elif num:
        files = files[:num]
    return files


def predict_files(files, out_dir, agents=("car", "robot"), v0_car=8.0, v0_robot=1.0,
                  car_speeds=None, modes=3, plot=True, verbose=True):
    """Run motion prediction on 2.5D map files; returns one record per frame.

    car_speeds: optional {frame_id: speed m/s} (e.g. from KITTI OXTS); frames missing
    from it fall back to v0_car.
    """
    os.makedirs(out_dir, exist_ok=True)
    predictors = {name: MotionPredictor(AGENTS[name]()) for name in agents}
    summary = []

    for f in files:
        fid = os.path.basename(f).split("_")[0]
        m25 = load_25d(f)
        v0 = {"car": (car_speeds or {}).get(fid, v0_car), "robot": v0_robot}
        results, record = [], dict(frame=fid, agents={})
        for name, pred in predictors.items():
            t0 = time.perf_counter()
            res = pred.predict(m25, v0=v0[name], n_modes=modes)
            ms = (time.perf_counter() - t0) * 1000
            results.append(res)
            best = res["modes"][0]["states"]
            record["agents"][name] = dict(
                v0=round(float(v0[name]), 3),
                start_z=float(pred.agent.start_z),
                runtime_ms=round(ms, 1),
                emergency_stop=res["emergency_stop"],
                goal=[round(v, 2) for v in res["route"]["goal"]],
                goal_reachable=res["route"]["reachable_goal"],
                feasible_rollouts=res["n_feasible"],
                total_rollouts=res["n_candidates"],
                occupancy=res["grid"].meta,
                modes=[mode_to_json(m, res["dt"]) for m in res["modes"]],
            )
            if verbose:
                print(f"[{fid}] {name:5s} {ms:6.0f} ms | v0 {v0[name]:4.1f} | feasible {res['n_feasible']:4d}/{res['n_candidates']}"
                      f" | goal ({res['route']['goal'][0]:+.1f}, {res['route']['goal'][1]:.1f})"
                      f" | best end ({best[-1, 0]:+.1f}, {best[-1, 1]:.1f}) v={best[-1, 3]:.1f}"
                      f" p={res['modes'][0]['prob']:.2f}" + ("  EMERGENCY STOP" if res["emergency_stop"] else ""))

        with open(os.path.join(out_dir, f"{fid}_motion.json"), "w") as fh:
            json.dump(record, fh, indent=1)
        if plot:
            from visualize import save_figure
            save_figure(results, fid, os.path.join(out_dir, f"{fid}_motion.png"))
        summary.append(record)
    return summary


def main():
    args = parse_args()
    files = list_maps(args.data_dir, args.frames, args.num)
    if not files:
        sys.exit(f"No 2.5D maps found in {args.data_dir}")
    car_speeds = None
    if args.speeds_json:
        with open(args.speeds_json) as fh:
            car_speeds = {k: float(v) for k, v in json.load(fh).items()}
    summary = predict_files(files, args.out_dir, args.agents, args.v0_car, args.v0_robot,
                            car_speeds, args.modes, plot=not args.no_plot)
    n = len(summary)
    for name in args.agents:
        em = sum(r["agents"][name]["emergency_stop"] for r in summary)
        rt = np.mean([r["agents"][name]["runtime_ms"] for r in summary])
        print(f"{name}: {n} frames, {em} emergency stops, mean runtime {rt:.0f} ms")
    print(f"Outputs in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
