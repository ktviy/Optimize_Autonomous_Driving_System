import math

import torch

def get_next_multiple_of_16(val):
    return math.ceil(val / 16) * 16

def voxel_to_25d(point_cloud_3d, res, x_range, y_range, z_range):
    x, y, z = point_cloud_3d[:, 0], point_cloud_3d[:, 1], point_cloud_3d[:, 2]

    mask = (x >= x_range[0]) & (x < x_range[1]) & (z >= z_range[0]) & (z < z_range[1])
    x, y, z = x[mask], y[mask], z[mask]

    # Calculate the grid dimensions 
    grid_w = get_next_multiple_of_16(int((x_range[1] - x_range[0]) / res))
    grid_h = get_next_multiple_of_16(int((z_range[1] - z_range[0]) / res))

    # Calculate the voxel indices for each point in the point cloud and make sure they are within the grid bounds
    idx_x = torch.clamp(torch.floor((x - x_range[0]) / res).long(), 0, grid_w - 1)
    idx_z = torch.clamp(torch.floor((z - z_range[0]) / res).long(), 0, grid_h - 1)

    flat_indices = idx_x + idx_z * grid_w
    num_cells = grid_w * grid_h 

    y_min_grid = torch.full((num_cells,), float('inf'), device=point_cloud_3d.device)
    y_max_grid = torch.full((num_cells,), float('-inf'), device=point_cloud_3d.device)

    y_min_grid = y_min_grid.scatter_reduce(0, flat_indices, y, reduce='amin', include_self=False)
    y_max_grid = y_max_grid.scatter_reduce(0, flat_indices, y, reduce='amax', include_self=False)

    y_min_grid[y_min_grid == float('inf')] = 0
    y_max_grid[y_max_grid == float('-inf')] = 0

    # Normalize the y_min and y_max values to the range [0, 1] based on the y_range
    y_range_span = y_range[1] - y_range[0]
    y_min_norm = (y_min_grid - y_range[0]) / y_range_span
    y_max_norm = (y_max_grid - y_range[0]) / y_range_span

    y_min_norm = torch.clamp(y_min_norm, 0.0, 1.0)
    y_max_norm = torch.clamp(y_max_norm, 0.0, 1.0)

    y_min_grid = y_min_norm.view(1, grid_h, grid_w) # [1, H, W]
    y_max_grid = y_max_norm.view(1, grid_h, grid_w) # [1, H, W]
    
    #[B, C, H, W] with B=1, C=2 (y_min and y_max), H=grid_h, W=grid_w
    output = torch.cat([y_min_grid, y_max_grid], dim=0).unsqueeze(0) 

    return output

