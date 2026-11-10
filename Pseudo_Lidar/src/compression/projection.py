# Project a 2D point into a 3D point 
import torch 

def depth_to_3d_vector(depth_tensor, f, B, cx, cy):
    device = depth_tensor.device
    H, W = depth_tensor.shape

    Z = depth_tensor

    v_coords, u_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')

    X = ((u_coords - cx) * Z) / f
    Y = ((v_coords - cy) * Z) / f

    point_cloud = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)

    return point_cloud

