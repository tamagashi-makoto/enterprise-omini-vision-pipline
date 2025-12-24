
import os

target_file = "/Users/ah000277/upwork/enterprise-omini-vision-pipline/omni_vision/venv/lib/python3.13/site-packages/sam3/train/data/sam3_image_dataset.py"

print(f"Reading {target_file}...")
with open(target_file, "r") as f:
    content = f.read()

# Replace the specific import line
old_import = "from decord import cpu, VideoReader"
new_import = """try:
    from decord import cpu, VideoReader
except ImportError:
    # Mocking decord for image-only usage (Mac compatibility)
    cpu = None
    VideoReader = None
"""

if old_import in content:
    print("Found 'decord' import. Patching...")
    new_content = content.replace(old_import, new_import)
    with open(target_file, "w") as f:
        f.write(new_content)
    print("Patch applied.")
else:
    print("Import line not found. Already patched?")

