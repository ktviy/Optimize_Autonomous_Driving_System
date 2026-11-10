# Running the FSD pipeline on KITTI (school lab)

Pipeline: stereo images → SDN depth (Pseudo-LiDAR++) → 2.5D BEV map → occupancy grid → car/robot motion prediction → evaluation.
One command runs all of it: `run_pipeline.py`. It skips finished stages, so you can stop and restart it.

Everything here was tested on synthetic KITTI-format data (`tools/make_synthetic_kitti.py`), both object and tracking modes, on CPU. Nothing was tested on a real GPU or on real KITTI images, so do step 4 (the quick test) first.

---

## 1. What to download from KITTI

Go to <https://www.cvlibs.net/datasets/kitti/> (free account). You only need the **training** parts. After unzipping you can delete the `testing/` folders to save space.

### A. Object benchmark (main evaluation: depth + occupancy)
Page: *3D Object* → downloads

| File | Size | Needed for |
|---|---|---|
| `data_object_image_2.zip` (left colour) | 12 GB | **required** |
| `data_object_image_3.zip` (right colour) | 12 GB | **required** (stereo) |
| `data_object_calib.zip` | 16 MB | **required** |
| `data_object_label_2.zip` | 5 MB | occupancy recall, SNN training |
| `data_object_velodyne.zip` | 29 GB | optional: depth error vs LiDAR |

### B. Tracking benchmark (motion evaluation: real speed + real future path)
Page: *Tracking* → downloads

| File | Size | Needed for |
|---|---|---|
| `data_tracking_image_2.zip` | 15 GB | **required** |
| `data_tracking_image_3.zip` | 15 GB | **required** (stereo) |
| `data_tracking_calib.zip` | 1 MB | **required** |
| `data_tracking_oxts.zip` | 8 MB | **required** for ADE/FDE (GPS/IMU) |
| `data_tracking_label_2.zip` | 9 MB | occupancy recall |
| `data_tracking_velodyne.zip` | 35 GB | not used |

If you are short on disk, do A without velodyne (≈24 GB) first, then B.

## 2. Folder layout

Put the data next to the `FSD` folder (the scripts look there by default):

```
<somewhere>/
├── FSD/                         <- this project
└── Dataset/
    ├── KITTI/
    │   └── training/
    │       ├── image_2/  000000.png ...
    │       ├── image_3/
    │       ├── calib/    000000.txt ...
    │       ├── label_2/          (optional)
    │       └── velodyne/         (optional)
    └── KITTI_tracking/
        └── training/
            ├── image_02/0000/000000.png ...
            ├── image_03/0000/
            ├── calib/0000.txt ...
            ├── oxts/0000.txt ...
            └── label_02/0000.txt ...  (optional)
```

Other locations work: pass the path with `--kitti-root` / `--tracking-root`.

## 3. Environment

```bash
python -m venv venv
venv\Scripts\activate          # Linux: source venv/bin/activate
nvidia-smi                     # note the CUDA version the driver supports
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match the driver
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # must print True
```

The SDN checkpoint must be at `Pseudo_Lidar/results/sdn_kitti_train_set/checkpoint.pth` (it is in the project folder).

## 4. Quick test first (5 minutes)

Run all commands from inside `FSD/`.

```bash
python run_pipeline.py object --kitti-root ../Dataset/KITTI --out runs/check --check
python run_pipeline.py object --kitti-root ../Dataset/KITTI --out runs/quick --max-frames 5
```

`--check` only checks the setup: Python packages, checkpoint, GPU, missing images/calib, labels/velodyne, disk space. Lines starting with `PROBLEM` must be fixed. Lines starting with `note` are only information.

The 5-frame run should end with `finished; outputs in runs/quick` and write `runs/quick/summary.md`.

## 5. Full runs

### Object, validation subset (500 frames; the same frames as the earlier results)
```bash
python run_pipeline.py object --kitti-root ../Dataset/KITTI --split Pseudo_Lidar/split/subval.txt --out runs/object_subval
```
Use the full held-out set with `--split Pseudo_Lidar/split/val.txt` (3769 frames).
**Do not evaluate on `train.txt`**: the SDN checkpoint was trained on those frames.

### Tracking, one sequence (e.g. 0001)
```bash
python run_pipeline.py tracking --tracking-root ../Dataset/KITTI_tracking --seq 0001 --out runs/tracking_0001
```
This stages the sequence into the object layout, converts the colon-less tracking calib, reads the real ego speed from OXTS, and scores the predicted car path against the real path driven (ADE/FDE over 4 s).
Good sequences to try: 0001, 0004, 0011, 0020. Frames in the last 4 s of a sequence have no full future path, so they are skipped in ADE/FDE.

### Useful flags
| Flag | Meaning |
|---|---|
| `--max-frames N` | only the first N frames |
| `--stages depth,bev,motion,eval` | default. Add `snn` to also export SNN spike features |
| `--batch 2` | lower this if the GPU runs out of memory |
| `--no-plot` | skip PNG figures (faster, less disk) |
| `--force` | redo stages whose outputs already exist |
| `--device cpu` | no GPU (≈10 s per frame for depth) |
| `--dynamic_bs` | use the stereo baseline from calib instead of 0.54 m |

## 6. What you get (`runs/<name>/`)

| Path | Content |
|---|---|
| `depth/000123.npy` | SDN depth map (metres, 375×1242) |
| `bev25d/000123_25d.npy` | 2.5D BEV map [1,2,512,208] (same format as `Pseudo_Lidar/results/processed_25d`) |
| `motion/000123_motion.json/.png` | occupancy + 3 predicted trajectories with probabilities, for car and robot |
| `summary.md` / `summary.json` | all metrics, below |
| `motion_errors_per_frame.json`, `oxts_speeds.json` | tracking mode only |
| `sdn_log/` | depth network log |

Metrics in `summary.md`:
- **depth_vs_lidar**: MAE / RMSE / abs_rel / δ1, and MAE per range 0–20, 20–35, 35–50 m. Needs velodyne.
- **occupancy_recall**: share of labelled objects (Car, Van, Truck, Pedestrian, Cyclist, ...) inside the BEV area, not heavily occluded or truncated, that have ≥3 obstacle cells in their footprint. Given per class and per range. Needs labels.
- **motion_vs_oxts** (tracking): ADE/FDE of the most likely mode, minADE/minFDE over the 3 modes, and the constant-velocity baseline `cv_ade`/`cv_fde`.
- **planner_car / planner_robot**: emergency stops, planned stops, runtime, obstacle share.

Expected scale (from the synthetic test): depth MAE ≈0.15 m at 0–20 m, ≈1.5 m at 20–35 m, ≈5 m at 35–50 m. Recall is high close up and drops beyond 35 m.

## 7. Retraining the classifiers (optional)

The SNN and Transformer classify a 2.5D map by the **first object in its KITTI label**. Produce maps first (step 5), then:
```bash
cd SNN_T
python train.py --data-dir ../runs/object_subval/bev25d --label-dir ../../Dataset/KITTI/training/label_2 --epochs 50 --T 1
cd ../Transformer
python train_t.py --data-dir ../runs/object_subval/bev25d --label-dir ../../Dataset/KITTI/training/label_2 --epochs 10
```
Both print the majority-class baseline accuracy; compare against it. `--T 4` gives the SNN 4 time steps (the old model effectively used 1).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | reinstall torch with the CUDA index URL matching `nvidia-smi` |
| `CUDA out of memory` | `--batch 1` |
| `image_03 ... missing` | the right-camera zip is not extracted (stereo needs both) |
| `No module named spikingjelly.clock_driven` | fine with spikingjelly 2.x (handled), or `pip install spikingjelly==0.0.0.0.14` |
| spikingjelly 0.0.0.0.14 import error about tensorboard | `pip install tensorboard` |
| `_pickle.UnpicklingError` / `weights_only` | already handled; update to this code version |
| Run interrupted | rerun the same command; finished frames are skipped |
| Many emergency stops | look at `motion/*.png`; stereo noise near the car makes false walls |
