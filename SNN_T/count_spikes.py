#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib

try:
    import tkinter
    matplotlib.use('TkAgg') # Thử dùng backend hỗ trợ hiển thị cửa sổ
except Exception:
    matplotlib.use('Agg', force=True) # Fallback về Agg nếu không có môi trường đồ họa

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import logging
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train import KITTI25DDataset, Spikformer, set_surrogate, find_label_dir
from model.sj_compat import LIF_TYPES, functional, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Count spikes and plot spike stats')
    parser.add_argument('--data-dir', required=False, default=os.path.join(SCRIPT_DIR, '..', 'Pseudo_Lidar', 'results', 'processed_25d'), help='Path to processed 2.5D npy files')
    parser.add_argument('--label-dir', required=False, default=None, help='Optional label directory')
    parser.add_argument('--checkpoint', default=os.path.join(SCRIPT_DIR, 'weights', 'checkpoint_011.pth'), help='Path to checkpoint .pth')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--out-dir', default=os.path.join(SCRIPT_DIR, 'results'))
    parser.add_argument('--no-show', action='store_true', help='do not open a plot window')
    parser.add_argument('--debug-first-batch', action='store_true', help='Process only the first batch and print diagnostics')
    parser.add_argument('--set-threshold', type=float, default=0.5, help='Optionally set all LIF v_threshold to this value for testing')
    return parser.parse_args()


def load_model(device, checkpoint_path=None):
    model = Spikformer(
        img_size_h=512,
        img_size_w=208,
        num_classes=11,
        in_channels=2,
        depths=8,
        embed_dims=64,
        num_heads=1
    )
    set_surrogate(model)
    model.to(device)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ck = load_checkpoint(checkpoint_path, map_location=device)
        if 'model' in ck:
            model.load_state_dict(ck['model'])
        else:
            model.load_state_dict(ck)
    return model


def robust_to_numpy(x):
    try:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.array(x)


class SimpleNpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.file_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        x = torch.from_numpy(data).float()
        return x, torch.tensor(0, dtype=torch.long)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    if args.label_dir is None:
        dataset = SimpleNpyDataset(args.data_dir)
    else:
        try:
            dataset = KITTI25DDataset(args.data_dir, args.label_dir)
        except FileNotFoundError:
            dataset = SimpleNpyDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = load_model(device, args.checkpoint)
    model.eval()

    logging.getLogger().setLevel(logging.ERROR)

    spike_records = []
    hooks = []
    rec_inp_records = {}

    def make_hook(name):
        def hook(module, inp, out):
            pre = None
            try:
                for elt in inp:
                    if torch.is_tensor(elt):
                        pre = elt.detach().cpu()
                        break
            except Exception:
                pre = None
            spike_records.append((name, out))
            if pre is not None:
                rec_inp_records.setdefault(name, []).append(pre)
        return hook

    for i, m in enumerate(model.modules()):
        if isinstance(m, LIF_TYPES):
            h = m.register_forward_hook(make_hook(f"LIF_{i}"))
            hooks.append(h)

    lif_map = {}
    for i, m in enumerate(model.modules()):
        if isinstance(m, LIF_TYPES):
            lif_map[f"LIF_{i}"] = m

    if args.set_threshold is not None:
        for name, m in lif_map.items():
            if getattr(m, 'v_threshold', None) is not None:
                m.v_threshold = float(args.set_threshold)
        print(f"Set all LIF v_threshold to {args.set_threshold}")

    sample_spike_counts = []
    spike_per_neuron_list = [] # Lưu trực tiếp giá trị trung bình per-neuron của từng sample
    time_spike_accum = None
    MAX_SAMPLES = 200 

    with torch.no_grad():
        for x, _ in loader:
            spike_records.clear()
            x = x.permute(1, 0, 2, 3, 4).to(device).float()
            
            if x.shape[0] != args.T:
                t0 = x.shape[0]
                if t0 == 0:
                    continue
                reps = math.ceil(args.T / t0)
                x = x.repeat(reps, 1, 1, 1, 1)[:args.T]

            functional.reset_net(model)
            _ = model(x)

            B = x.shape[1]
            batch_total = np.zeros(B, dtype=np.float64)
            batch_time_accum = np.zeros(args.T, dtype=np.float64)
            
            # Lưu tổng số spike và tổng số nơ-ron riêng cho từng sample trong batch này
            sample_spikes_per_sample = np.zeros(B, dtype=np.float64)
            sample_neurons_per_sample = np.zeros(B, dtype=np.float64)

            for name, rec in spike_records:
                rec_tensor = None
                if torch.is_tensor(rec):
                    rec_tensor = rec
                elif isinstance(rec, (tuple, list)):
                    for elt in rec:
                        if torch.is_tensor(elt):
                            rec_tensor = elt
                            break
                
                arr = robust_to_numpy(rec_tensor) if rec_tensor is not None else robust_to_numpy(rec)
                if arr.size == 0:
                    continue
                
                if arr.ndim >= 2 and arr.shape[0] == args.T:
                    per_time_layer = arr.sum(axis=tuple(range(1, arr.ndim)))
                    if len(per_time_layer) == args.T:
                        batch_time_accum += per_time_layer

                    if arr.shape[1] == B:
                        # 1. Tổng spike của layer này cho từng sample trong batch (Sum tất cả trừ axis 1)
                        sum_axes = tuple(i for i in range(arr.ndim) if i != 1)
                        per_sample_layer = arr.sum(axis=sum_axes)
                        if len(per_sample_layer) == B:
                            batch_total += per_sample_layer[:B]
                            sample_spikes_per_sample += per_sample_layer[:B]
                            
                            # 2. Số lượng nơ-ron của layer này (C * H * W * T)
                            if arr.ndim > 2:
                                layer_neurons = np.prod(arr.shape[2:]) * args.T
                                sample_neurons_per_sample += layer_neurons

            sample_spike_counts.extend(batch_total.tolist())
            
            # Tính trung bình spike per neuron cho từng sample ngay trong batch
            with np.errstate(divide='ignore', invalid='ignore'):
                per_sample_avg = np.where(
                    sample_neurons_per_sample > 0,
                    sample_spikes_per_sample / sample_neurons_per_sample,
                    0
                )
            spike_per_neuron_list.extend(per_sample_avg.tolist())

            if time_spike_accum is None:
                time_spike_accum = batch_time_accum
            else:
                time_spike_accum += batch_time_accum

            functional.reset_net(model)
            if args.debug_first_batch:
                break
            if len(sample_spike_counts) >= MAX_SAMPLES:
                sample_spike_counts = sample_spike_counts[:MAX_SAMPLES]
                spike_per_neuron_list = spike_per_neuron_list[:MAX_SAMPLES]
                print(f"[INFO] Đã đạt giới hạn {MAX_SAMPLES} mẫu, dừng sớm.")
                break

    # Chuyển sang numpy array để vẽ
    sample_spike_counts = np.array(sample_spike_counts)
    spike_per_neuron = np.array(spike_per_neuron_list)
    if time_spike_accum is not None:
        time_spike_accum = np.array(time_spike_accum)




    # --- PHẦN VẼ 2 BIỂU ĐỒ ---
    fig, axs = plt.subplots(2, 1, figsize=(6, 9))
    
    # 1. Biểu đồ trên: Spike trung bình trên mỗi nơ-ron theo sample
    if spike_per_neuron.size > 0:
        axs[0].hist(spike_per_neuron, bins=50, color='C0')
        axs[0].set_xlabel('Spike count per neuron')
        axs[0].set_ylabel('Number of samples')
    else:
        axs[0].text(0.5, 0.5, 'No spike data recorded', ha='center')
        axs[0].set_axis_off()

    # 2. Biểu đồ dưới: Tổng số spike theo time-step (bắt đầu từ 0)
    if time_spike_accum is not None:
        tvals = np.arange(len(time_spike_accum)) 
        axs[1].bar(tvals, time_spike_accum, color='C1', width=0.6)
        axs[1].set_title('Total spikes per time-step')
        axs[1].set_xlabel('Time step')
        axs[1].set_ylabel('Total spikes')

        axs[1].set_xticks(tvals)
        axs[1].set_xticklabels([int(t) for t in tvals])
        axs[1].set_xlim(-0.5, len(tvals) - 0.5)
    else:
        axs[1].text(0.5, 0.5, 'No per-time spike data', ha='center')
        axs[1].set_axis_off()

    # Cấu hình hiển thị số đầy đủ, không dùng ký hiệu 1e7 cho cả 2 biểu đồ
    for ax in axs:
        formatter = matplotlib.ticker.ScalarFormatter()
        formatter.set_scientific(False)
        formatter.set_useOffset(False)
        ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    
    # Lưu file ảnh
    out_path = os.path.abspath(os.path.join(args.out_dir, 'spike_stats.png'))
    fig.savefig(out_path, dpi=300)
    print(f"[INFO] Successfully saved plot to: {out_path}")

    # Hiển thị biểu đồ lên màn hình
    try:
        if not args.no_show:
            plt.show()
    except Exception as e:
        print(f"[INFO] Không thể bật cửa sổ hiển thị đồ họa trực tiếp: {e}")

    plt.close(fig)


if __name__ == '__main__':
    main()