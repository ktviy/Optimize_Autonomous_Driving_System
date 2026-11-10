# Optimize Autonomous Driving System

An end-to-end, camera-only perception and planning pipeline for autonomous driving, evaluated on the KITTI benchmark.

```
Stereo images ─► SDN depth ─► 2.5D BEV map ─► Occupancy grid ─► Motion prediction ─► Evaluation
 (KITTI)        (Pseudo-LiDAR++)  (compression)   (SNN / Transformer)  (Motion_Planning)    (pipeline/)
```

A single entry point, [`run_pipeline.py`](run_pipeline.py), runs every stage. Finished stages are skipped, so a run can be stopped and resumed.

## Repository structure

Folders are listed in the order the data flows through the pipeline.

```
.
├── Pseudo_Lidar/          # Stage 1 – Stereo depth (SDN, Pseudo-LiDAR++) and 3D → 2.5D BEV compression
│   ├── src/               #   depth network, dataloaders, compression (projection, voxel_to_25d, map_visual)
│   ├── gdc/               #   graph-based depth correction
│   ├── split/             #   KITTI train / val / subval frame lists
│   └── results/           #   SDN checkpoint (checkpoint.pth, Git LFS)
├── SNN_T/                 # Stage 2a – Spiking Neural Network (Spikformer) occupancy model
│   ├── model/             #   network + SpikingJelly compatibility layer
│   ├── configs/
│   ├── Occupancy map/     #   inference, 4D → static occupancy conversion, visualisation
│   ├── weights/           #   training checkpoints (Git LFS)
│   └── results/, output_maps_static/   # sample outputs
├── Transformer/           # Stage 2b – Dense Transformer baseline for comparison with the SNN
│   ├── model/
│   └── weights/           #   training checkpoints (Git LFS)
├── Motion_Planning/       # Stage 3 – BEV occupancy, agents (car / robot), multi-modal trajectory planner
│   └── outputs/           #   sample motion predictions (JSON + PNG)
├── pipeline/              # Stage 4 – KITTI I/O and evaluation metrics (depth, recall, ADE/FDE)
├── tools/                 # Utilities: synthetic KITTI generator, checkpoint summary
├── docs/                  # Lab guide (setup, dataset download, full runs, troubleshooting)
├── logs/                  # RAM-footprint benchmark and training summaries
├── run_pipeline.py        # One-command pipeline entry point
└── requirements.txt
```

## Quick start

```bash
python -m venv venv
venv\Scripts\activate                 # Linux/macOS: source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA driver
pip install -r requirements.txt
git lfs pull                          # fetch model checkpoints
```

Check the setup, then run a 5-frame smoke test:

```bash
python run_pipeline.py object --kitti-root ../Dataset/KITTI --out runs/check --check
python run_pipeline.py object --kitti-root ../Dataset/KITTI --out runs/quick --max-frames 5
```

No KITTI data yet? Generate a synthetic KITTI-format dataset with `tools/make_synthetic_kitti.py`.

## Full runs

| Mode | Command |
|---|---|
| Object (500-frame validation subset) | `python run_pipeline.py object --kitti-root ../Dataset/KITTI --split Pseudo_Lidar/split/subval.txt --out runs/object_subval` |
| Tracking (one sequence, ADE/FDE vs OXTS) | `python run_pipeline.py tracking --tracking-root ../Dataset/KITTI_tracking --seq 0001 --out runs/tracking_0001` |

Results land in `runs/<name>/summary.md`. See [docs/LAB_GUIDE.md](docs/LAB_GUIDE.md) for dataset layout, all flags, metrics and troubleshooting.

## Metrics

- **Depth vs LiDAR** – MAE / RMSE / abs_rel / δ1 by range (0–20, 20–35, 35–50 m)
- **Occupancy recall** – share of labelled objects covered by obstacle cells, per class and range
- **Motion vs OXTS** – ADE / FDE, minADE / minFDE over 3 modes, constant-velocity baseline
- **Planner** – emergency stops, planned stops, runtime, obstacle share

## Not included in the repository

Large generated data is excluded and can be regenerated with `run_pipeline.py`:
SDN depth maps, processed 2.5D maps, full per-frame motion outputs, and the virtual environment.

## Acknowledgements

The depth stage builds on [Pseudo-LiDAR++](https://github.com/mileyan/Pseudo_Lidar_V2) (You et al., ICLR 2020); see [`Pseudo_Lidar/LICENSE`](Pseudo_Lidar/LICENSE). Spiking layers use [SpikingJelly](https://github.com/fangwei123456/spikingjelly).
