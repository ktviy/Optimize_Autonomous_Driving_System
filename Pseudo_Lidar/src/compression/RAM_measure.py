import torch

def generate_benchmark_table(dense_3d_tensor, compressed_25d_tensor):
    """
    Calculate and print a memory comparison table in Markdown format
    """
    # Calculate dimensions (bytes -> MB)
    mem_3d_mb = (dense_3d_tensor.element_size() * dense_3d_tensor.nelement()) / (1024**2)
    mem_25d_mb = (compressed_25d_tensor.element_size() * compressed_25d_tensor.nelement()) / (1024**2)
    
    # Create Markdown table
    markdown_table = (
        f"| Model Variant | RAM Footprint (MB) | Savings (%) |\n"
        f"|:---|:---|:---|\n"
        f"| Dense 3D | {mem_3d_mb:.2f} | 0% |\n"
        f"| 2.5D Optimized | {mem_25d_mb:.2f} | {((mem_3d_mb - mem_25d_mb)/mem_3d_mb)*100:.1f}% |"
    )
    return markdown_table