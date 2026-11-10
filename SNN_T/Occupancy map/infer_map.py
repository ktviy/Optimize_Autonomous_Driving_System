import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so local imports work when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.model import Spikformer
from model.sj_compat import load_checkpoint
import argparse


class SimpleNpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.file_list = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        data = np.load(file_path)
        # Standardize shape to [C, H, W] for a single static frame
        if data.ndim == 4:
            data = data[0]  # Take first batch if shape is [B, C, H, W] or [1, C, H, W]
        elif data.ndim == 5:
            data = data[0, 0]  # Take first batch and first time step if [B, T, C, H, W]
            
        x = torch.from_numpy(data.astype(np.float32))
        return x, torch.tensor(0, dtype=torch.long), file_path

# --- Data paths (override with --weights / --data-dir / --out-dir / --max-frames) ---
_p = argparse.ArgumentParser(description='Export SNN spike-feature maps for 2.5D maps')
_p.add_argument('--weights', default=os.path.join(ROOT, 'weights', 'checkpoint_011.pth'))
_p.add_argument('--data-dir', default=os.path.join(ROOT, '..', 'Pseudo_Lidar', 'results', 'processed_25d'))
_p.add_argument('--out-dir', default=os.path.join(ROOT, 'output_maps_static'))
_p.add_argument('--max-frames', type=int, default=10, help='0 = all frames')
_args, _ = _p.parse_known_args()
WEIGHTS_PATH = _args.weights
DATA_DIR = _args.data_dir
OUTPUT_DIR = _args.out_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trained_model(weights_path, device):
    model = Spikformer(
        img_size_h=512,
        img_size_w=208,
        num_classes=11,
        in_channels=2,
        depths=8,
        embed_dims=64,
        num_heads=1
    )
   
    checkpoint = load_checkpoint(weights_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
       
    model.to(device)
    model.eval()
    return model

def extract_static_occupancy_features():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sử dụng SimpleNpyDataset hoặc tùy chỉnh dataset trả về kèm đường dẫn/file độc lập
    dataset = SimpleNpyDataset(DATA_DIR)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = load_trained_model(WEIGHTS_PATH, device)
    model.eval()

    def extract_features_frame(model, frame_tensor):
        """
        Extract spatial features for a single static frame tensor of shape [B, C, H, W].
        Returns a tensor of shape [B, Cf, Hf, Wf].
        """
        frame = frame_tensor
        try:
            tmp = frame.unsqueeze(0)  # [T=1, B, C, H, W]
            patch_embed = getattr(model, 'patch_embed')
            block = getattr(model, 'block')
            feat, (Hf, Wf) = patch_embed(tmp)
            for blk in block:
                feat = blk(feat)
            feat = feat.squeeze(0)  # [B, Cf, Hf, Wf]
            return feat
        except Exception:
            hook_out = {}
            def make_hook(name):
                def hook(module, inp, out):
                    hook_out['feat'] = out.detach()
                return hook

            last_conv = None
            for mn, m in reversed(list(model.named_modules())):
                if isinstance(m, torch.nn.Conv2d):
                    last_conv = (mn, m)
                    break

            if last_conv is not None:
                name, module_conv = last_conv
                h = module_conv.register_forward_hook(make_hook(name))
                try:
                    _ = model(frame.unsqueeze(0))  # Pass [1, B, C, H, W]
                    feat = hook_out.get('feat', None)
                except Exception:
                    feat = None
                finally:
                    h.remove()

                if feat is not None:
                    if feat.dim() == 5:
                        feat = feat.squeeze(0)
                    return feat

            raise RuntimeError('Could not extract spatial features for a single static frame')

    with torch.no_grad():
        for idx, item in enumerate(data_loader):
            # Handle dataset outputs depending on whether file_path is returned
            if len(item) == 3:
                x, target, file_path = item
                # DataLoader collates non-tensor items into lists/tuples; extract single element
                if isinstance(file_path, (list, tuple)) and len(file_path) > 0:
                    file_path = file_path[0]
                # If file_path is bytes, decode to str
                if isinstance(file_path, bytes):
                    file_path = file_path.decode('utf-8')
            else:
                x, target = item
                file_path = dataset.file_list[idx]

            x = x.to(device).float()  # Shape: [B, C, H, W] where B=1

            # Đảm bảo tensor đúng dạng [B, C, H, W] cho single-frame static map
            if x.dim() == 5:
                x = x.squeeze(1) # Bỏ chiều T nếu lỡ có
            
            # Trích xuất đặc trưng tĩnh cho đúng một khung hình độc lập này
            feat_b = extract_features_frame(model, x)  # [B, Cf, Hf, Wf]
            
            # Lưu tensor dưới dạng static map (không cần chiều T > 1)
            # Shape lưu: [B, Cf, Hf, Wf] hoặc [Cf, Hf, Wf]
            tensor_static = feat_b.cpu()
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(OUTPUT_DIR, f"occupancy_static_{base_name}.npy")
            np.save(output_path, tensor_static.numpy())
            
            print(f"Exported Static Occupancy Map for {base_name} with shape: {tensor_static.shape}")

            if _args.max_frames and idx + 1 >= _args.max_frames:
                break

if __name__ == '__main__':
    extract_static_occupancy_features()