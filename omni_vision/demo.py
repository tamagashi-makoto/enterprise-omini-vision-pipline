"""
Integrated Omni-Vision Pipeline Demo
=====================================

This demo shows the complete vision pipeline in action:
1. YOLOv12: Fast screening detection
2. RF-DETR: High-precision detection for complex scenes
3. Florence-2: Text-prompted open-vocabulary detection
4. SAM3: Segmentation based on detection results

The pipeline automatically selects the best model based on scene complexity
and generates segmentation masks for detected objects.
"""
import asyncio
import sys
from pathlib import Path
import urllib.request
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from PIL import Image, ImageDraw
import torch
import numpy as np


# ============================================================
# Configuration
# ============================================================
SAMPLE_IMAGES = {
    "street": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
    "office": "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800",
}

DENSITY_THRESHOLD = 15  # Switch to RF-DETR if more than this many detections


# ============================================================
# Helper Functions
# ============================================================
def download_sample_images(output_dir: Path):
    """Download sample images for testing."""
    output_dir.mkdir(exist_ok=True)
    images = {}
    
    for name, url in SAMPLE_IMAGES.items():
        filepath = output_dir / f"{name}.jpg"
        if not filepath.exists():
            print(f"  Downloading {name} image...")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                print(f"  Failed: {e}, creating placeholder")
                img = Image.new('RGB', (800, 600), color='gray')
                img.save(filepath)
        images[name] = filepath
    
    return images


def draw_detections(image: Image.Image, detections: list, color="red"):
    """Draw bounding boxes on image."""
    draw = ImageDraw.Draw(image)
    for det in detections:
        box = det.box
        draw.rectangle(box, outline=color, width=2)
        label = f"{det.label}: {det.confidence:.0%}"
        draw.text((box[0], box[1] - 15), label, fill=color)
    return image


# ============================================================
# Pipeline Demo
# ============================================================
class OmniVisionDemo:
    """Integrated demo of the Omni-Vision pipeline."""
    
    def __init__(self):
        self.yolo = None
        self.rf_detr = None
        self.florence2 = None
        self.sam3 = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    async def load_models(self, models_to_load=None):
        """Load specified models (or all if not specified)."""
        from src.model_wrappers import (
            YOLOv12Wrapper, 
            RFDETRWrapper, 
            Florence2Wrapper, 
            SAM3Wrapper
        )
        
        if models_to_load is None:
            models_to_load = ["yolo", "rfdetr", "sam3"]  # Skip florence2 by default
        
        print("\n📦 Loading models...")
        
        if "yolo" in models_to_load:
            print("  • YOLOv12...")
            self.yolo = YOLOv12Wrapper()
            await self.yolo.load()
        
        if "rfdetr" in models_to_load:
            print("  • RF-DETR...")
            self.rf_detr = RFDETRWrapper()
            await self.rf_detr.load()
        
        if "florence2" in models_to_load:
            print("  • Florence-2...")
            self.florence2 = Florence2Wrapper()
            await self.florence2.load()
        
        if "sam3" in models_to_load:
            print("  • SAM3...")
            self.sam3 = SAM3Wrapper()
            await self.sam3.load()
        
        print("✅ Models loaded!\n")
    
    async def run_detection_pipeline(self, image: Image.Image):
        """
        Run the adaptive detection pipeline:
        1. YOLOv12 for fast screening
        2. If high density, switch to RF-DETR for precision
        """
        print("🔍 Stage 1: Fast Screening with YOLOv12...")
        start = time.time()
        yolo_detections = await self.yolo.predict(image)
        yolo_time = time.time() - start
        
        print(f"   Found {len(yolo_detections)} objects in {yolo_time:.2f}s")
        
        # Check if scene is complex (high density)
        if len(yolo_detections) > DENSITY_THRESHOLD:
            print(f"\n🔬 Stage 2: High-density scene detected! Switching to RF-DETR...")
            start = time.time()
            rf_detections = await self.rf_detr.predict(image, threshold=0.5)
            rf_time = time.time() - start
            
            print(f"   Found {len(rf_detections)} objects in {rf_time:.2f}s")
            return rf_detections, "RF-DETR"
        else:
            print("   Scene density is low, using YOLO results.")
            return yolo_detections, "YOLOv12"
    
    async def run_text_detection(self, image: Image.Image, query: str):
        """
        Run text-prompted detection with SAM3.
        """
        print(f"\n🔤 Text-prompted detection: '{query}'")
        
        if self.sam3:
            print("   Using SAM3 for text-prompted segmentation...")
            start = time.time()
            result = await self.sam3.predict(image, text_prompt=query)
            sam_time = time.time() - start
            
            masks = result.get("masks", [])
            mask_count = len(masks) if hasattr(masks, '__len__') else 0
            print(f"   Generated {mask_count} masks in {sam_time:.2f}s")
            return result
        else:
            print("   SAM3 not loaded, skipping segmentation.")
            return {}
    
    async def run_segmentation(self, image: Image.Image, detections: list):
        """
        Run SAM3 segmentation on detected objects.
        Uses text prompts based on detected class labels.
        """
        if not self.sam3 or not detections:
            return {}
        
        # Get unique labels from detections
        labels = list(set([d.label for d in detections]))
        
        print(f"\n🎭 Stage 3: Segmentation with SAM3...")
        print(f"   Segmenting objects: {', '.join(labels[:5])}{'...' if len(labels) > 5 else ''}")
        
        # Use the most common label for text-prompted segmentation
        from collections import Counter
        label_counts = Counter([d.label for d in detections])
        most_common = label_counts.most_common(1)[0][0]
        
        start = time.time()
        result = await self.sam3.predict(image, text_prompt=most_common)
        sam_time = time.time() - start
        
        masks = result.get("masks", [])
        mask_count = len(masks) if hasattr(masks, '__len__') else 0
        print(f"   Generated {mask_count} masks for '{most_common}' in {sam_time:.2f}s")
        
        return result
    
    async def run_full_pipeline(self, image: Image.Image, text_query: str = None):
        """
        Run the complete pipeline:
        1. Detection (YOLO → RF-DETR if needed)
        2. Segmentation (SAM3)
        
        If text_query is provided, use SAM3 for text-prompted segmentation.
        """
        print("\n" + "="*60)
        print("🚀 OMNI-VISION PIPELINE")
        print("="*60)
        
        total_start = time.time()
        
        if text_query:
            # Text-prompted mode: Use SAM3 directly
            result = await self.run_text_detection(image, text_query)
            detections = []
            model_used = "SAM3 (text-prompted)"
        else:
            # Standard mode: Detection → Segmentation
            detections, model_used = await self.run_detection_pipeline(image)
            result = await self.run_segmentation(image, detections)
        
        total_time = time.time() - total_start
        
        # Summary
        print("\n" + "-"*60)
        print("📊 PIPELINE SUMMARY")
        print("-"*60)
        print(f"   Detection model: {model_used}")
        print(f"   Objects detected: {len(detections)}")
        
        masks = result.get("masks", []) if result else []
        mask_count = len(masks) if hasattr(masks, '__len__') else 0
        print(f"   Masks generated: {mask_count}")
        print(f"   Total time: {total_time:.2f}s")
        print("="*60 + "\n")
        
        return {
            "detections": detections,
            "model_used": model_used,
            "masks": masks,
            "total_time": total_time
        }


# ============================================================
# Main Demo
# ============================================================
async def main():
    print("\n" + "="*60)
    print("🔮 OMNI-VISION INTEGRATED PIPELINE DEMO")
    print("="*60)
    
    # System info
    print(f"\n📌 System Info:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Download test images
    print(f"\n📥 Preparing test images...")
    output_dir = Path("/app/test_images")
    images = download_sample_images(output_dir)
    
    # Initialize pipeline
    demo = OmniVisionDemo()
    
    # Load models (skip florence2 due to compatibility issues)
    await demo.load_models(["yolo", "rfdetr", "sam3"])
    
    # ============================================================
    # Demo 1: Standard Detection → Segmentation Pipeline
    # ============================================================
    print("\n" + "="*60)
    print("📸 DEMO 1: Standard Detection Pipeline (Street Scene)")
    print("="*60)
    
    street_image = Image.open(images["street"]).convert("RGB")
    print(f"   Image size: {street_image.size}")
    
    result1 = await demo.run_full_pipeline(street_image)
    
    # Show top detections
    if result1["detections"]:
        print("   Top detections:")
        for det in result1["detections"][:5]:
            print(f"      • {det.label}: {det.confidence:.1%}")
    
    # ============================================================
    # Demo 2: Text-prompted Segmentation
    # ============================================================
    print("\n" + "="*60)
    print("📸 DEMO 2: Text-prompted Segmentation")
    print("="*60)
    
    result2 = await demo.run_full_pipeline(street_image, text_query="person")
    
    # ============================================================
    # Demo 3: Compare YOLO vs RF-DETR
    # ============================================================
    print("\n" + "="*60)
    print("📸 DEMO 3: Model Comparison (YOLO vs RF-DETR)")
    print("="*60)
    
    print("\n🔹 YOLOv12:")
    start = time.time()
    yolo_results = await demo.yolo.predict(street_image)
    yolo_time = time.time() - start
    print(f"   Detections: {len(yolo_results)}, Time: {yolo_time:.2f}s")
    
    print("\n🔹 RF-DETR:")
    start = time.time()
    rfdetr_results = await demo.rf_detr.predict(street_image, threshold=0.5)
    rfdetr_time = time.time() - start
    print(f"   Detections: {len(rfdetr_results)}, Time: {rfdetr_time:.2f}s")
    
    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE")
    print("="*60)
    print("""
The Omni-Vision pipeline demonstrated:
  1. Adaptive detection: YOLO for speed, RF-DETR for precision
  2. Text-prompted segmentation with SAM3
  3. Seamless model switching based on scene complexity
    """)


if __name__ == "__main__":
    asyncio.run(main())
