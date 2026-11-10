import os
import glob
import numpy as np

src_dir = os.path.join(os.path.dirname(__file__), '..', 'output_maps')
out_dir = os.path.join(os.path.dirname(__file__), '..', 'output_maps_static')
os.makedirs(out_dir, exist_ok=True)

matches = sorted(glob.glob(os.path.join(src_dir, 'occupancy_4d_tensor_*.npy')))
print('Found', len(matches), '4D tensors')
for path in matches:
    arr = np.load(path)
    base = os.path.splitext(os.path.basename(path))[0]
    # Expect arr shape [T, C, H, W] or [B, T, C, H, W]
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        print('Skipping', path, 'unexpected shape', arr.shape)
        continue
    T = arr.shape[0]
    for t in range(T):
        single = arr[t]
        out_name = f"occupancy_static_{base}_t{t}.npy"
        out_path = os.path.join(out_dir, out_name)
        np.save(out_path, single)
        print('Wrote', out_path)

print('Done')
