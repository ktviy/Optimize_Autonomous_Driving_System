import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Occupancy4DMapProcessor:
    def __init__(self, threshold=0.5, grid_resolution=0.5):
        self.threshold = threshold
        self.grid_resolution = grid_resolution

    def convert_tensor_to_occupancy_map(self, tensor_path):
        tensor_data = np.load(tensor_path)
        if tensor_data.ndim == 4:
            tensor_data = tensor_data[0]

        if tensor_data.ndim == 3:
            C, H, W = tensor_data.shape
            tensor_torch = torch.from_numpy(tensor_data.astype(np.float32))
            eps = 1e-6
            normed = tensor_torch.clone()
            for c in range(C):
                sl = tensor_torch[c]
                m = float(sl.mean())
                s = float(sl.std())
                if s < eps:
                    normed[c] = sl - m
                else:
                    normed[c] = (sl - m) / (s + eps)

            prob_map = torch.sigmoid(normed * 1.5).numpy()
            occ = (prob_map > self.threshold).astype(np.int8)

            cleaned = occ.copy()
            for c in range(C):
                grid = occ[c]
                pad = ((1, 1), (1, 1))
                pg = np.pad(grid, pad_width=pad, mode='constant', constant_values=0)
                neigh_sum = np.zeros_like(pg)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        neigh_sum += np.roll(np.roll(pg, dy, axis=0), dx, axis=1)
                neigh = neigh_sum[1:-1, 1:-1]
                keep = (grid == 1) & (neigh >= 2)
                cleaned[c] = keep.astype(np.int8)

            return np.expand_dims(cleaned, axis=0), np.expand_dims(prob_map, axis=0)

        raise RuntimeError(f"Unsupported tensor shape for static map: {tensor_data.shape}")

    def get_occupied_coordinates_over_time(self, occupancy_map_3d):
        T = occupancy_map_3d.shape[0]
        trajectory_coords = {}

        for t in range(T):
            grid = occupancy_map_3d[t]
            if grid.ndim == 3:
                C_dim, H_dim, W_dim = grid.shape
                indices = np.argwhere(grid == 1)
                if indices.size == 0:
                    coords = np.zeros((0, 3), dtype=np.float32)
                else:
                    c = indices[:, 0].astype(np.float32)
                    h = indices[:, 1].astype(np.float32)
                    w = indices[:, 2].astype(np.float32)
                    
                    x = (w - W_dim / 2.0) * self.grid_resolution
                    y = h * self.grid_resolution
                    z = c * self.grid_resolution
                    coords = np.stack([x, y, z], axis=1)
            else:
                H_dim, W_dim = grid.shape
                indices = np.argwhere(grid == 1)
                if indices.size == 0:
                    coords = np.zeros((0, 3), dtype=np.float32)
                else:
                    h = indices[:, 0].astype(np.float32)
                    w = indices[:, 1].astype(np.float32)
                    x = (w - W_dim / 2.0) * self.grid_resolution
                    y = h * self.grid_resolution
                    z = np.zeros_like(x)
                    coords = np.stack([x, y, z], axis=1)

            trajectory_coords[t] = coords

        return trajectory_coords


class ProbabilisticOccupancyGrid3D:
    def __init__(self, x_range=(-5, 5), y_range=(0, 30), z_range=(-2, 5), resolution=0.5):
        self.resolution = resolution
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

        self.nx = int((self.x_max - self.x_min) / resolution)
        self.ny = int((self.y_max - self.y_min) / resolution)
        self.nz = int((self.z_max - self.z_min) / resolution)

        self.log_odds_map = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        
        # Hệ số Log-odds chuẩn cho Ray Casting
        self.l_occ = 0.85   # Giá trị tăng khi bị chiếm chỗ
        self.l_free = -0.4  # Giá trị giảm khi tia đi qua (không gian trống)
        self.l_min = -5.0 
        self.l_max = 5.0 

    def world_to_grid(self, coords):
        ix = np.floor((coords[:, 0] - self.x_min) / self.resolution).astype(np.int32)
        iy = np.floor((coords[:, 1] - self.y_min) / self.resolution).astype(np.int32)
        iz = np.floor((coords[:, 2] - self.z_min) / self.resolution).astype(np.int32)
        return ix, iy, iz

    def bresenham_3d(self, x0, y0, z0, x1, y1, z1):
        """Thuật toán Bresenham 3D tìm các voxel trên đường truyền tia từ cảm biến đến vật cản"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        
        xs = 1 if x1 > x0 else -1
        ys = 1 if y1 > y0 else -1
        zs = 1 if z1 > z0 else -1
        
        # Driving axis is X
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            x, y, z = x0, y0, z0
            line = [(x, y, z)]
            while x != x1:
                x += xs
                if p1 >= 0:
                    y += ys
                    p1 -= 2 * dx
                p1 += 2 * dy
                if p2 >= 0:
                    z += zs
                    p2 -= 2 * dx
                p2 += 2 * dz
                if x == x1 and y == y1 and z == z1:
                    break
                line.append((x, y, z))
            return line

        # Driving axis is Y
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            x, y, z = x0, y0, z0
            line = [(x, y, z)]
            while y != y1:
                y += ys
                if p1 >= 0:
                    x += xs
                    p1 -= 2 * dy
                p1 += 2 * dx
                if p2 >= 0:
                    z += zs
                    p2 -= 2 * dy
                p2 += 2 * dz
                if x == x1 and y == y1 and z == z1:
                    break
                line.append((x, y, z))
            return line

        # Driving axis is Z
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            x, y, z = x0, y0, z0
            line = [(x, y, z)]
            while z != z1:
                z += zs
                if p1 >= 0:
                    y += ys
                    p1 -= 2 * dz
                p1 += 2 * dy
                if p2 >= 0:
                    x += xs
                    p2 -= 2 * dz
                p2 += 2 * dx
                if x == x1 and y == y1 and z == z1:
                    break
                line.append((x, y, z))
            return line

    def update_map(self, occupied_coords):
        if len(occupied_coords) == 0:
            return
        
        # Giả lập vị trí cảm biến (Sensor origin) đặt tại tâm dưới của xe trong hệ tọa độ grid
        sensor_world = np.array([[0.0, 0.0, 0.0]])
        sx, sy, sz = self.world_to_grid(sensor_world)
        x0, y0, z0 = int(sx[0]), int(sy[0]), int(sz[0])

        ix, iy, iz = self.world_to_grid(occupied_coords)
        
        for i in range(len(ix)):
            x1, y1, z1 = int(ix[i]), int(iy[i]), int(iz[i])
            
            # Kiểm tra điểm đích nằm trong bản đồ
            if not (0 <= x1 < self.nx and 0 <= y1 < self.ny and 0 <= z1 < self.nz):
                continue

            # Lấy danh sách các voxel dọc theo tia truyền bằng Bresenham
            ray_voxels = self.bresenham_3d(x0, y0, z0, x1, y1, z1)
            
            for rx, ry, rz in ray_voxels:
                if 0 <= rx < self.nx and 0 <= ry < self.ny and 0 <= rz < self.nz:
                    # Các ô dọc đường đi là không gian trống (Free space) -> giảm log-odds
                    self.log_odds_map[rx, ry, rz] += self.l_free

            # Ô cuối cùng chứa vật cản -> tăng log-odds
            self.log_odds_map[x1, y1, z1] += self.l_occ

        # Giới hạn giá trị log-odds trong khoảng min/max
        self.log_odds_map = np.clip(self.log_odds_map, self.l_min, self.l_max)

    def get_occupancy_grid_states(self, prob_threshold=0.5):
        prob_map = 1.0 / (1.0 + np.exp(-self.log_odds_map))
        grid_state = np.full(self.log_odds_map.shape, -1, dtype=np.int8)
        
        # Ngưỡng phân định rõ ràng Occupied (1), Free (0), Unknown (-1)
        grid_state[prob_map > (0.5 + prob_threshold * 0.3)] = 1
        grid_state[prob_map < (0.5 - prob_threshold * 0.3)] = 0
        
        # --- BỘ LỌC KHỬ NHIỄU 3D (Denoising 3D Morphology) ---
        pad_grid = np.pad(grid_state, pad_width=1, mode='constant', constant_values=-1)
        cleaned_grid = np.full_like(grid_state, -1)
        
        nx, ny, nz = grid_state.shape
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if grid_state[x, y, z] == 1:
                        neighbor_block = pad_grid[x:x+3, y:y+3, z:z+3]
                        occ_count = np.sum(neighbor_block == 1) - 1
                        if occ_count >= 2:
                            cleaned_grid[x, y, z] = 1
                        else:
                            cleaned_grid[x, y, z] = 0 
                    else:
                        cleaned_grid[x, y, z] = grid_state[x, y, z]
                        
        return cleaned_grid, prob_map


def visualize_occupancy_grid(grid_states, resolution=0.5, x_range=(-5, 5), y_range=(0, 30), z_range=(-2, 5), save_path=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')

    voxel_array = (grid_states == 1)
    
    if not np.any(voxel_array):
        print("Không có ô nào bị chiếm chỗ (Occupied) để hiển thị sau khi lọc nhiễu!")
        return

    nx, ny, nz = voxel_array.shape
    x_vals = np.linspace(x_range[0], x_range[1], nx + 1)
    y_vals = np.linspace(y_range[0], y_range[1], ny + 1)
    z_vals = np.linspace(z_range[0], z_range[1], nz + 1)
    
    colors = np.empty(voxel_array.shape, dtype=object)
    cmap = plt.get_cmap('viridis')
    z_min_val, z_max_val = z_range
    
    for z in range(nz):
        z_val = z * resolution + z_min_val
        norm_z = (z_val - z_min_val) / (z_max_val - z_min_val + 1e-6)
        color_val = cmap(np.clip(norm_z, 0.0, 1.0))
        color_tuple = tuple([float(x) for x in color_val])
        arr = np.empty((nx, ny), dtype=object)
        arr.fill(color_tuple)
        colors[:, :, z] = arr

    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    ax.voxels(X, Y, Z, voxel_array, facecolors=colors, edgecolors=None, alpha=0.8)

    ax.set_xlabel('X Axis (m) [Left/Right]')
    ax.set_ylabel('Y Axis (m) [Forward]')
    ax.set_zlabel('Z Axis (m) [Height]')
    ax.set_title(f'3D Voxel Occupancy Grid with Ray Casting (Voxels: {np.sum(voxel_array)})')
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    ax.view_init(elev=25, azim=-55)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    processor = Occupancy4DMapProcessor(threshold=0.5)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_maps_dir = os.path.join(base_dir, 'output_maps_static')
    
    sample_tensor_path = None
    if os.path.isdir(output_maps_dir):
        matches = sorted(glob.glob(os.path.join(output_maps_dir, 'occupancy_static_*.npy')))
        if matches:
            sample_tensor_path = matches[0] 

    if sample_tensor_path and os.path.exists(sample_tensor_path):
        matches = sorted(glob.glob(os.path.join(output_maps_dir, '*000048*.npy')))
        if not matches:
            print(f"No matching static tensor file for '000048' found in {output_maps_dir}.")
        else:
            images_out = os.path.join(output_maps_dir, 'images')
            sample = matches[0]
            print(f"Processing static tensor with Ray Casting: {sample}")
            occ_map_3d, prob_3d = processor.convert_tensor_to_occupancy_map(sample)
            coords_by_time = processor.get_occupied_coordinates_over_time(occ_map_3d)

            grid_map_3d = ProbabilisticOccupancyGrid3D(x_range=(-5, 5), y_range=(0, 30), z_range=(-2, 5), resolution=0.5)
            for t, coords in coords_by_time.items():
                if len(coords) > 0:
                    grid_map_3d.update_map(coords)

            final_grid_states, probability_voxel_map = grid_map_3d.get_occupancy_grid_states()

            print(f"Shape của Voxel Grid Map: {final_grid_states.shape}")
            print(f"Số lượng ô Occupied (1) sau Ray Casting & Lọc nhiễu: {np.sum(final_grid_states == 1)}")

            base = os.path.splitext(os.path.basename(sample))[0]
            out_img = os.path.join(images_out, f"{base}_raycasting.png")
            visualize_occupancy_grid(final_grid_states,
                                    resolution=grid_map_3d.resolution,
                                    x_range=(grid_map_3d.x_min, grid_map_3d.x_max),
                                    y_range=(grid_map_3d.y_min, grid_map_3d.y_max),
                                    z_range=(grid_map_3d.z_min, grid_map_3d.z_max),
                                    save_path=out_img)