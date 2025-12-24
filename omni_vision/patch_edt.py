
import os
import torch
import cv2
import numpy as np

# Path to the file to patch
target_file = "/Users/ah000277/upwork/enterprise-omini-vision-pipline/omni_vision/venv/lib/python3.13/site-packages/sam3/model/edt.py"

# New content
new_content = """import torch
import cv2
import numpy as np

def edt_triton(data: torch.Tensor):
    \"\"\"
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images using OpenCV fallback.
    Replacing Triton implementation for macOS compatibility.
    
    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.
    Returns:
        A tensor of the same shape as data containing the EDT.
    \"\"\"
    # Ensure data is 3D
    if data.dim() != 3:
        raise ValueError("Data must be 3D (B, H, W)")
        
    B, H, W = data.shape
    device = data.device
    
    # Move to CPU for OpenCV
    # Convert properly: data is likely float or bool. OpenCV expects uint8.
    # Note: cv2.distanceTransform calculates distance to closest ZERO pixel.
    # If data is logic mask (1=object), we want distance to 0.
    data_cpu = data.detach().cpu().numpy()
    if data_cpu.dtype != np.uint8:
        data_cpu = data_cpu.astype(np.uint8)
    
    outputs = []
    for i in range(B):
        img = data_cpu[i]
        # cv2.DIST_L2, mask size 5
        dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        outputs.append(dist)
        
    outputs = np.array(outputs, dtype=np.float32)
    
    # Convert back to tensor and device
    return torch.from_numpy(outputs).to(device)
"""

print(f"Patching {target_file}...")
with open(target_file, "w") as f:
    f.write(new_content)
print("Patch applied successfully.")
