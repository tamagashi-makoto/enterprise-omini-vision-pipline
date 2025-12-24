
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.real_wrappers import YOLOv12RealWrapper, Florence2Wrapper, SAM3RealWrapper, RFDETRRealWrapper
from src.config import Config
import traceback
import sys
from transformers import PreTrainedModel

# Monkey Patch for Florence-2
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False 


async def test_models():
    print("--- Starting Model Verification ---")

    # 1. Test YOLOv12
    print("\n[1/4] Testing YOLOv12...")
    try:
        yolo = YOLOv12RealWrapper(Config.PATH_YOLO)
        await yolo.load()
        if yolo.model:
            print("SUCCESS: YOLOv12 loaded.")
        else:
            print("FAILURE: YOLOv12 did not load.")
    except Exception as e:
        print(f"FAILURE: YOLOv12 Exception: {e}")

    # 2. Test Florence-2
    print("\n[2/4] Testing Florence-2...")
    try:
        florence = Florence2Wrapper(Config.PATH_FLORENCE_2)
        await florence.load()
        if florence.model:
            print("SUCCESS: Florence-2 loaded.")
        else:
            print("FAILURE: Florence-2 did not load.")
    except Exception as e:
        print(f"FAILURE: Florence-2 Exception: {e}")
        traceback.print_exc()

    # 3. Test SAM3
    print("\n[3/4] Testing SAM3...")
    try:
        # Assumes sam3.pt or weights are handled by the library/config
        sam3 = SAM3RealWrapper(Config.PATH_SAM_3) 
        await sam3.load()
        if sam3.model:
            print("SUCCESS: SAM3 loaded.")
        else:
            print("FAILURE: SAM3 did not load.")
    except Exception as e:
        print(f"FAILURE: SAM3 Exception: {e}")
        traceback.print_exc()

    # 4. Test RF-DETR (API check only)
    print("\n[4/4] Testing RF-DETR Wrapper (Init only)...")
    try:
        rf = RFDETRRealWrapper()
        await rf.load()
        print("SUCCESS: RF-DETR Wrapper initialized.")
    except Exception as e:
        print(f"FAILURE: RF-DETR Exception: {e}")

import traceback
import sys

if __name__ == "__main__":
    # Force print to stdout
    try:
        asyncio.run(test_models())
    except:
        traceback.print_exc(file=sys.stdout)


