#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import logging
import random
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from spikingjelly.activation_based import monitor
from torch.utils.data import DataLoader, Dataset
from model.model import Spikformer, spikformer
from model.sj_compat import LIF_TYPES, functional, surrogate
from timm.scheduler import create_scheduler
from timm.utils import setup_default_logging

# --- Configuration (all paths relative to this repo; override on the command line) ---
FSD_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')
DEFAULT_DATA_DIR = os.path.join(FSD_ROOT, 'Pseudo_Lidar', 'results', 'processed_25d')


def find_label_dir(kitti_training_dir=None):
    """KITTI labels live in training/label_2 (official zip) or training/label/label_2 (older layout)."""
    roots = [kitti_training_dir] if kitti_training_dir else [
        os.environ.get('KITTI_TRAINING', ''),
        os.path.join(FSD_ROOT, '..', 'Dataset', 'KITTI', 'training'),
        os.path.join(FSD_ROOT, 'Dataset', 'KITTI', 'training'),
    ]
    for r in roots:
        for sub in ('label_2', os.path.join('label', 'label_2')):
            d = os.path.join(r, sub)
            if r and os.path.isdir(d):
                return os.path.abspath(d)
    return None


def get_next_checkpoint_path(weights_dir, prefix="checkpoint", extension=".pth"):
    existing_files = [
        f for f in os.listdir(weights_dir)
        if f.startswith(prefix) and f.endswith(extension)
    ]
    indices = []
    for filename in existing_files:
        stem = os.path.splitext(filename)[0]
        suffix = stem[len(prefix):]
        if suffix.startswith("_"):
            suffix = suffix[1:]
        if suffix.isdigit():
            indices.append(int(suffix))

    next_index = max(indices, default=-1) + 1
    return os.path.join(weights_dir, f"{prefix}_{next_index:03d}{extension}")


class KITTI25DDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        self.label_dir = label_dir

    def __len__(self):
        return len(self.file_list)

    def _get_label_from_filename(self, filename):
        file_id = os.path.basename(filename).split('_')[0]
        label_path = os.path.join(self.label_dir, f"{file_id}.txt")
        
        class_mapping = {
            'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 
            'Van': 3, 'Truck': 4, 'Person_sitting': 5, 
            'Tram': 6, 'Misc': 7
        }

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                class_name = parts[0]
                
                # Ignore 'DontCare'
                if class_name == 'DontCare':
                    continue
                
                # Trả về ID của vật thể đầu tiên tìm thấy (ví dụ: Car, Pedestrian...)
                return class_mapping.get(class_name, 7)
        
        return 7 # If no valid label is found, return 'Misc' class ID

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        x = torch.from_numpy(data).float()
        # stored as [1, 2, H, W]; keep a leading time axis of length 1 -> [T=1, 2, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        target = self._get_label_from_filename(self.file_list[idx])

        # If no valid label is found, set target to 0 (Misc class)
        if target == -1 or target is None:
            target = 0

        return x, torch.tensor(target, dtype=torch.long)


def to_time_major(x, T):
    """[B, T0, C, H, W] -> [T, B, C, H, W]; a static map (T0 = 1) is repeated T times."""
    x = x.permute(1, 0, 2, 3, 4)
    if T > x.shape[0]:
        x = x.repeat(int(np.ceil(T / x.shape[0])), 1, 1, 1, 1)[:T]
    return x

# 1. Surrogate Gradient
surrogate_function = surrogate.ATan() 

def set_surrogate(model):
    for m in model.modules():
        if hasattr(m, 'surrogate_function'):
            m.surrogate_function = surrogate_function
            # Ensure the backend handles the surrogate properly
            if hasattr(m, 'v_threshold'):
                m.v_threshold = 0.5 # As defined in SSA layer

_logger = logging.getLogger('train')

def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch SNN 25D Training')
    parser.add_argument('--model', default='spikformer', type=str)
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR, help='Path to processed 2.5D npy files')
    parser.add_argument('--label-dir', type=str, default=None,
                        help='KITTI label_2 folder (default: auto-detect ../Dataset/KITTI/training/label_2 or $KITTI_TRAINING)')
    parser.add_argument('--weights-dir', type=str, default=WEIGHTS_DIR)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--T', type=int, default=1,
                        help='SNN time steps. The shipped checkpoints were trained with T=1 '
                             '(the static map was never repeated); T>1 repeats the map T times.')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0, help='fixes the train/val split and init')

    parser.add_argument('--sched', default='cosine', type=str, help='Scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    
    return parser.parse_args()

def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, target in dataloader:
            x, target = to_time_major(x, getattr(model, 'T', 1)).to(device).float(), target.to(device)
            output = model(x)
            output_mean = output.mean(dim=0)
            _, predicted = output_mean.max(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            functional.reset_net(model)
    return correct / total

def train_one_epoch(model, dataloader, optimizer, criterion, device, T):
    model.train()
    total_loss = 0
    spike_seq_monitor = monitor.OutputMonitor(model, LIF_TYPES)
    for x, target in dataloader:
        # x shape from loader: [B, T0, C, H, W] -> [T, B, C, H, W]
        x, target = to_time_major(x, T).to(device).float(), target.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        T, B, C = output.shape
        loss = criterion(output.reshape(T * B, C), target.repeat_interleave(T))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
    
    # Calculate mean firing rate for logging
    fr = 0
    if len(spike_seq_monitor.records) > 0:
        fr = 0
        for sn in spike_seq_monitor.records:
            fr += float(sn.float().mean())
        fr /= len(spike_seq_monitor.records)
        _logger.info(f"Mean Firing Rate: {fr:.4f}")
    else:
        _logger.info("Mean Firing Rate: Monitor did not record any spikes.")

    spike_seq_monitor.remove_hooks()

    return total_loss / len(dataloader)

def main():
    setup_default_logging()
    args = _parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    label_dir = args.label_dir or find_label_dir()
    if label_dir is None:
        raise SystemExit('KITTI label_2 folder not found: pass --label-dir or set KITTI_TRAINING')
    os.makedirs(args.weights_dir, exist_ok=True)

    # Load Dataset
    dataset = KITTI25DDataset(args.data_dir, label_dir)
    if len(dataset) < 2:
        raise SystemExit(f'Need at least 2 maps in {args.data_dir}, found {len(dataset)}')

    # Split dataset into training and validation sets (seeded -> reproducible)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    # Accuracy must beat "always predict the most common class" to mean anything
    val_labels = [dataset._get_label_from_filename(dataset.file_list[i]) for i in val_dataset.indices]
    counts = Counter(val_labels)
    majority = counts.most_common(1)[0][1] / len(val_labels)
    _logger.info(f"Val class counts: {dict(counts)} | majority-class baseline accuracy: {majority*100:.2f}%")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model
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
    model.T = args.T
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_acc = 0.0
    best_checkpoint = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.T)
        val_acc = validate(model, val_loader, device)

        if lr_scheduler: lr_scheduler.step(epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc,
                'majority_baseline': majority,
                'epoch': epoch,
                'T': args.T,
            }

        _logger.info(f"Epoch {epoch} | Loss: {train_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

    if best_checkpoint is None:
        best_checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': None,
            'val_acc': best_acc,
            'epoch': args.epochs - 1,
        }

    checkpoint_path = get_next_checkpoint_path(args.weights_dir, prefix="checkpoint")
    torch.save(best_checkpoint, checkpoint_path)
    _logger.info(f"Saved checkpoint to {checkpoint_path}")

if __name__ == '__main__':
    main()