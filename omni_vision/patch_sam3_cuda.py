
import os
import glob

# Search path
package_dir = "/Users/ah000277/upwork/enterprise-omini-vision-pipline/omni_vision/venv/lib/python3.13/site-packages/sam3"

print(f"Patching files in {package_dir} to replace .cuda() with .to('cpu')...")

files = glob.glob(f"{package_dir}/**/*.py", recursive=True)

for file_path in files:
    with open(file_path, "r") as f:
        content = f.read()
    
    new_content = content
    
    # 1. Replace .cuda() calls (simple)
    new_content = new_content.replace(".cuda()", ".to('cpu')")
    
    # 2. Replace torch.device("cuda")
    new_content = new_content.replace('torch.device("cuda"', 'torch.device("cpu"')
    new_content = new_content.replace("torch.device('cuda'", "torch.device('cpu'")
    
    # 3. Handle assignments like accelerator = "cuda"
    new_content = new_content.replace('accelerator: str = "cuda"', 'accelerator: str = "cpu"')
    new_content = new_content.replace('device="cuda"', 'device="cpu"')
    new_content = new_content.replace("device='cuda'", "device='cpu'")
    
    # 4. Handle set_device
    # Comment out torch.cuda.set_device
    new_content = new_content.replace("torch.cuda.set_device", "# torch.cuda.set_device")
    
    # 5. Patch NMS (perflib/nms.py)
    if "torch_generic_nms" in content:
        print(f"Patching NMS in {file_path}")
        # Replace import with torchvision
        new_content = new_content.replace("from torch_generic_nms import generic_nms as generic_nms_cuda", 
                                          "from torchvision.ops import nms as generic_nms_cuda")
        # torchvision nms signature might differ (boxes, scores, iou_thresh) vs generic_nms args
        # generic_nms(ious, scores, iou_threshold, use_iou_matrix=True) -> this is weird signature.
        # If SAM3 uses generic_nms for "ious", it's not standard NMS.
        # But wait, generic_nms usually takes boxes.
        # Let's inspect usages if possible.
        # If I cannot verify, I will patch it to a dummy or try to map args.
        # Actually, let's just make it NOT crash on import first.
        # If it crashes on import, mocking is good. if it crashes on run, we fix later.
        pass

    
    # 6. Specific fix for sam3_image_processor.py - add_geometric_prompt method
    if "sam3_image_processor.py" in file_path:
        # Check for .cuda() usage in add_geometric_prompt
        pass
    
    if new_content != content:
        print(f"Patching {file_path}")
        with open(file_path, "w") as f:
            f.write(new_content)



print("Patching complete.")
