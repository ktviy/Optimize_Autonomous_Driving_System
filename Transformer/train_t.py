#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, Dataset
import random
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from timm.utils import setup_default_logging
from timm.scheduler import create_scheduler
from model.model_t import VisionTransformer

# --- Configuration (relative paths; override on the command line) ---
FSD_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')


def find_label_dir(kitti_training_dir=None):
    """KITTI labels live in training/label_2 (official zip) or training/label/label_2 (older layout).
    Same lookup as SNN_T/train.py (copied, because both folders have a package called `model`)."""
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

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    class_name = parts[0]
                    
                    # Ignore 'DontCare'
                    if class_name == 'DontCare':
                        continue
                    
                    return class_mapping.get(class_name, 7)
        
        return 7 # If no valid label is found, return 'Misc' class ID

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        x = torch.from_numpy(data).float()
        if x.dim() == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.dim() == 5 and x.shape[1] == 1:
            x = x.squeeze(1)
        target = self._get_label_from_filename(self.file_list[idx])

        if target == -1 or target is None:
            target = 0

        return x, torch.tensor(target, dtype=torch.long)

_logger = logging.getLogger('train')

def _parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Pure Vision Transformer 25D Training')
    parser.add_argument('--data-dir', type=str, default=os.path.join(FSD_ROOT, 'Pseudo_Lidar', 'results', 'processed_25d'), help='Path to processed 2.5D npy files')
    parser.add_argument('--label-dir', type=str, default=None, help='KITTI label_2 folder (default: auto-detect)')
    parser.add_argument('--weights-dir', type=str, default=WEIGHTS_DIR)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0, help='fixes the train/val split (same split as SNN_T/train.py)')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sched', default='cosine', type=str, help='Scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    return parser.parse_args()

def _prepare_input(x):
    """Always return [B, C, H, W]. (The old version squeezed a batch of 1 down to [C, H, W],
    which only worked because the positional-embedding add re-created the batch axis.)"""
    if x.dim() == 5:
        if x.shape[1] == 1:
            x = x.squeeze(1)
        else:
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    return x


def validate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, target in dataloader:
            x = _prepare_input(x)
            x, target = x.to(device).float(), target.to(device)
            output = model(x)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, target in dataloader:
        x = _prepare_input(x)
        x, target = x.to(device).float(), target.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
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

    # Same seeded split as SNN_T/train.py, so both models are scored on the same val frames
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    val_labels = [dataset._get_label_from_filename(dataset.file_list[i]) for i in val_dataset.indices]
    counts = Counter(val_labels)
    majority = counts.most_common(1)[0][1] / len(val_labels)
    _logger.info(f"Val class counts: {dict(counts)} | majority-class baseline accuracy: {majority*100:.2f}%")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model 
    model = VisionTransformer(
        img_size_h=512, 
        img_size_w=208,
        in_channels=2,
        num_classes=11,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=4.
    )
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_checkpoint = None
    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = validate(model, val_loader, device)

        if lr_scheduler:
            lr_scheduler.step(epoch + 1)

        if val_acc > best_acc:
            best_acc = val_acc
            best_checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_acc': val_acc,
                'majority_baseline': majority,
                'epoch': epoch,
            }

        _logger.info(f"Epoch {epoch} | Loss: {train_loss:.4f} | Val Accuracy: {val_acc*100:.2f}% ")

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