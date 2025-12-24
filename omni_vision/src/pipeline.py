"""
Omni-Vision Pipeline - Intelligent model orchestration.
"""
import asyncio
from typing import List, Dict, Any, Optional
from PIL import Image

from .config import Config, ModelType
from .model_wrappers import (
    YOLOv12Wrapper, 
    RFDETRWrapper, 
    Florence2Wrapper,  # Replaced DINOXWrapper
    SAM3Wrapper, 
    DetectionResult
)


class OmniVisionPipeline:
    """
    Intelligent orchestration pipeline that dynamically selects the best 
    computer vision model based on scene complexity and user intent.
    """

    def __init__(self):
        self.yolo = YOLOv12Wrapper()
        self.rf_detr = RFDETRWrapper()
        self.florence_2 = Florence2Wrapper()  # Replaced DINO-X
        self.sam_3 = SAM3Wrapper()
        self.models_loaded = False

    async def load_models(self):
        """Initializes and loads all models."""
        if not self.models_loaded:
            print("Loading all models...")
            await asyncio.gather(
                self.yolo.load(),
                self.rf_detr.load(),
                self.florence_2.load(),
                self.sam_3.load()
            )
            self.models_loaded = True
            print("All models loaded successfully!")

    async def analyze(
        self, 
        image: Image.Image, 
        text_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        
        Logic Flow:
        1. Stage 1 (Screening): Run YOLOv12.
        2. Stage 2 (Density Check): count > 15 -> Switch to RF-DETR.
        3. Stage 3 (Intent Check): if text_query -> Run Florence-2.
        4. Stage 4 (Segmentation): Run SAM 3 on final boxes.
        
        Args:
            image: PIL Image input.
            text_query: Optional text prompt for identifying specific objects.
            
        Returns:
             JSON-compatible dictionary with metadata, detections, and segmentation.
        """
        if not self.models_loaded:
            await self.load_models()

        active_mode = ModelType.YOLO_V12
        final_detections: List[DetectionResult] = []

        # --- Stage 1: Screening (YOLOv12) ---
        print("Stage 1: Running YOLOv12 screening...")
        yolo_results = await self.yolo.predict(image)
        final_detections = yolo_results
        active_mode = "YOLO-Fast"

        # --- Stage 2: Density Check ---
        if len(yolo_results) > Config.DENSITY_THRESHOLD:
            print(f"High density detected ({len(yolo_results)} objects). Switching to RF-DETR.")
            rf_results = await self.rf_detr.predict(image)
            final_detections = rf_results
            active_mode = "RF-DETR-High-Res"
        else:
            print(f"Density normal ({len(yolo_results)} objects). Keeping YOLOv12.")

        # --- Stage 3: Intent Check (Florence-2) ---
        if text_query:
            print(f"Text query received: '{text_query}'. Switching to Florence-2.")
            florence_results = await self.florence_2.predict(image, text_query=text_query)
            final_detections = florence_results
            active_mode = f"Florence-2 ({text_query})"

        # --- Stage 4: Segmentation (SAM 3) ---
        boxes = [d.box for d in final_detections]
        segmentation_result = {}
        segmentation_available = False
        
        if boxes:
            print(f"Stage 4: Running SAM3 segmentation on {len(boxes)} boxes...")
            segmentation_result = await self.sam_3.predict(
                image, 
                boxes=boxes,
                text_prompt=text_query
            )
            segmentation_available = True

        # Format Response
        response = {
            "meta": {
                "processing_mode": active_mode,
                "objects_detected": len(final_detections)
            },
            "detections": [d.to_dict() for d in final_detections],
            "masks_generated": len(segmentation_result.get("masks", [])),
            "segmentation_available": segmentation_available,
            "segmentation_scores": segmentation_result.get("scores", [])
        }
        
        return response
