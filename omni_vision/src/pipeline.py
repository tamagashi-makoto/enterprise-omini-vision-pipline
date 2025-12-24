import asyncio
from typing import List, Dict, Any, Optional, Tuple
from .config import Config, ModelType
from .model_wrappers import DetectionResult, ModelWrapper
from .real_wrappers import (
    YOLOv12RealWrapper,
    RFDETRRealWrapper,
    Florence2Wrapper,
    SAM3RealWrapper
)

class OmniVisionPipeline:
    def __init__(self):
        # Stage 1: YOLOv12 (Local)
        self.yolo = YOLOv12RealWrapper(Config.PATH_YOLO)
        
        # Stage 2: RF-DETR (Cloud API / Roboflow)
        self.rf_detr = RFDETRRealWrapper(model_id="rf-detr-v1") 
        
        # Stage 3: Florence-2 (Local HuggingFace)
        self.florence_2 = Florence2Wrapper(Config.PATH_FLORENCE_2)
        
        # Stage 4: SAM 3 (Local)
        self.sam_3 = SAM3RealWrapper(Config.PATH_SAM_3)
        
        self.models_loaded = False

    async def load_models(self):
        """Initializes and loads all models (simulated)."""
        if not self.models_loaded:
            # 1. YOLOv12
            try:
                await self.yolo.load()
            except Exception as e:
                print(f"Pipeline Warning: Failed to load YOLOv12: {e}")

            # 2. RF-DETR
            try:
                await self.rf_detr.load()
            except Exception as e:
                print(f"Pipeline Warning: Failed to load RF-DETR: {e}")

            # 3. Florence-2
            try:
                await self.florence_2.load()
            except Exception as e:
                print(f"Pipeline Warning: Failed to load Florence-2: {e}")

            # 4. SAM 3
            try:
                await self.sam_3.load()
            except Exception as e:
                print(f"Pipeline Warning: Failed to load SAM 3: {e}")

            self.models_loaded = True

    async def analyze(self, image: Any, text_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        
        Logic Flow:
        1. Stage 1 (Screening): Run YOLOv12.
        2. Stage 2 (Density Check): count > 15 -> Switch to RF-DETR.
        3. Stage 3 (Intent Check): if text_query -> Run DINO-X.
        4. Stage 4 (Segmentation): Run SAM 3 on final boxes.
        
        Args:
            image: Input image.
            text_query: Optional text prompt for identifying specific objects.
            
        Returns:
             JSON-compatible dictionary with metadata, detections, and segmentation status.
        """
        if not self.models_loaded:
            await self.load_models()

        active_mode = ModelType.YOLO_V12
        final_detections: List[DetectionResult] = []

        # --- Stage 1: Screening (YOLOv12) ---
        # Note: We always start with YOLO as a fast screener/default
        yolo_results = await self.yolo.predict(image)
        final_detections = yolo_results
        active_mode = "YOLO-Fast"

        # --- Stage 2: Density Check ---
        if len(yolo_results) > Config.DENSITY_THRESHOLD:
            # Too many objects, switch to high-precision model
            print(f"High density detected ({len(yolo_results)} objects). Switching to {ModelType.RF_DETR}.")
            rf_results = await self.rf_detr.predict(image)
            if rf_results:  # Only use RF-DETR if it actually returns results
                final_detections = rf_results
                active_mode = "RF-DETR-High-Res"
            else:
                print("RF-DETR returned no results. Keeping YOLO detections.")
                # final_detections stays as yolo_results
        else:
            print(f"Density normal ({len(yolo_results)} objects). Keeping {ModelType.YOLO_V12}.")

        # --- Stage 3: Intent Check ---
        # If user specifies what they are looking for, we prioritize that intent.
        # We assume Florence-2 is best for open-vocabulary queries.
        if text_query:
            print(f"Text query received: '{text_query}'. Switching to Florence-2.")
            florence_results = await self.florence_2.predict(image, text_query=text_query)
            final_detections = florence_results
            active_mode = f"Florence-2 ({text_query})"

        # --- Stage 4: Segmentation (SAM 3) ---
        # Generate masks for whatever bounding boxes we decided on.
        boxes = [d.box for d in final_detections]
        masks = []
        segmentation_available = False
        
        if boxes:
            print(f"Refining {len(boxes)} detections with SAM 3...")
            masks = await self.sam_3.predict(image, boxes=boxes)
            if masks:
                segmentation_available = True
            else:
                print("SAM 3 produced no masks.")

        # Format Response
        response = {
            "meta": {
                "processing_mode": active_mode,
                "objects_detected": len(final_detections)
            },
            "detections": [d.to_dict() for d in final_detections],
            "masks_generated": len(masks), # Metadata only, actual masks might be huge blobs
            "segmentation_available": segmentation_available
        }
        
        return response
