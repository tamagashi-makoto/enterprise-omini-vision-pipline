"""
Model wrappers for the Omni-Vision pipeline.
Implements real model inference with GPU support.
"""
import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from .config import Config, ModelType

# Weight file paths
WEIGHTS_DIR = Path(__file__).parent / "weights"


@dataclass
class DetectionResult:
    """Standardized output for detection models."""
    label: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]
    mask: Optional[np.ndarray] = None  # Optional segmentation mask

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "label": self.label,
            "confidence": self.confidence,
            "box": self.box
        }
        if self.mask is not None:
            result["has_mask"] = True
        return result


class ModelWrapper(ABC):
    """Abstract base class for all model wrappers."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
    
    @abstractmethod
    async def load(self):
        """Load the model weights."""
        pass

    @abstractmethod
    async def predict(self, image: Any, **kwargs) -> Any:
        """Run inference on the image."""
        pass


class YOLOv12Wrapper(ModelWrapper):
    """
    Wrapper for YOLOv12 (Screening Model).
    Uses ultralytics library with pretrained weights in src/weights/.
    """

    async def load(self):
        """Load YOLOv12 model from local weights."""
        from ultralytics import YOLO
        
        print(f"Loading {ModelType.YOLO_V12}... Device: {self.device}")
        
        # Check for local weights first
        weight_path = WEIGHTS_DIR / "yolo12m.pt"
        if weight_path.exists():
            self.model = YOLO(str(weight_path))
            print(f"Loaded YOLOv12 from local weights: {weight_path}")
        else:
            # Download pretrained model
            self.model = YOLO("yolo12m.pt")
            print("Loaded YOLOv12 from pretrained (downloading if needed)")
        
        # Move to GPU if available
        self.model.to(self.device)
        print(f"YOLOv12 ready on {self.device}")

    async def predict(self, image: Image.Image, **kwargs) -> List[DetectionResult]:
        """
        Run YOLOv12 object detection.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            List of detected objects with bounding boxes.
        """
        if self.model is None:
            await self.load()
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            lambda: self.model.predict(image, verbose=False)
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                label = result.names[cls]
                
                detections.append(DetectionResult(
                    label=label,
                    confidence=conf,
                    box=box
                ))
        
        return detections


class RFDETRWrapper(ModelWrapper):
    """
    Wrapper for RF-DETR using the official rfdetr package.
    Runs locally on GPU for high-precision object detection.
    """

    def __init__(self):
        super().__init__()

    async def load(self):
        """Load RF-DETR model."""
        from rfdetr import RFDETRBase
        
        print(f"Loading {ModelType.RF_DETR}... Device: {self.device}")
        
        self.model = RFDETRBase()
        
        # Note: optimize_for_inference() can cause TorchScript tracing errors
        # Skipping for now - inference still works without optimization
        # if self.device == "cuda":
        #     self.model.optimize_for_inference()
        
        print(f"RF-DETR ready on {self.device}")


    async def predict(self, image: Image.Image, threshold: float = 0.5, **kwargs) -> List[DetectionResult]:
        """
        Run RF-DETR detection.
        
        Args:
            image: PIL Image
            threshold: Confidence threshold (default: 0.5)
        
        Returns:
            List of high-precision detections.
        """
        if self.model is None:
            await self.load()
        
        # Import COCO classes for labels
        from rfdetr.util.coco_classes import COCO_CLASSES
        
        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        
        def _inference():
            return self.model.predict(image, threshold=threshold)
        
        result = await loop.run_in_executor(None, _inference)
        
        detections = []
        for i, (class_id, conf) in enumerate(zip(result.class_id, result.confidence)):
            box = result.xyxy[i].tolist()  # [x1, y1, x2, y2]
            label = COCO_CLASSES[int(class_id)]
            
            detections.append(DetectionResult(
                label=label,
                confidence=float(conf),
                box=box
            ))
        
        return detections


class Florence2Wrapper(ModelWrapper):
    """
    Wrapper for Florence-2 (Open-Vocabulary Detection).
    Uses Microsoft's Florence-2 model from HuggingFace.
    Replaces DINO-X as API key is not available.
    """

    def __init__(self):
        super().__init__()
        self.processor = None

    async def load(self):
        """Load Florence-2 model from HuggingFace."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print(f"Loading {ModelType.FLORENCE_2}... Device: {self.device}")
        
        if self.device == "cpu":
            print("WARNING: Florence-2 on CPU will be slow. GPU recommended.")
        
        model_id = "microsoft/Florence-2-base"
        
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager"  # Fix for transformers compatibility
        ).to(self.device)
        
        print(f"Florence-2 loaded on {self.device}")


    async def predict(self, image: Image.Image, text_query: str = "", **kwargs) -> List[DetectionResult]:
        """
        Run Florence-2 open-vocabulary detection.
        
        Args:
            image: PIL Image
            text_query: Text prompt describing what to detect
        
        Returns:
            List of detections matching the query.
        """
        if self.model is None:
            await self.load()
        
        if not text_query:
            return []
        
        # Florence-2 task for object detection with text
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = f"{task_prompt} {text_query}"
        
        loop = asyncio.get_event_loop()
        
        def _inference():
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            
            # Convert pixel_values to the model's dtype (float16 on CUDA)
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    do_sample=False,
                    num_beams=1  # Use greedy decoding to avoid beam search issues
                )
            
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            parsed = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            return parsed
        
        result = await loop.run_in_executor(None, _inference)
        
        detections = []
        if task_prompt in result:
            data = result[task_prompt]
            boxes = data.get("bboxes", [])
            labels = data.get("labels", [])
            
            for box, label in zip(boxes, labels):
                detections.append(DetectionResult(
                    label=label,
                    confidence=0.9,  # Florence-2 doesn't output confidence
                    box=list(box)
                ))
        
        return detections


class SAM3Wrapper(ModelWrapper):
    """
    Wrapper for SAM 3 (Segment Anything Model 3).
    Requires GPU - CPU execution is not supported.
    """

    def __init__(self):
        super().__init__()
        self.processor = None

    async def load(self):
        """Load SAM3 model onto GPU."""
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        print(f"Loading {ModelType.SAM_3}... Device: {self.device}")
        
        if self.device == "cpu":
            raise RuntimeError(
                "SAM3 requires GPU. CPU execution is not supported. "
                "Please ensure CUDA is available and properly configured."
            )
        
        # HuggingFace auth from environment
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        
        # Build SAM3 model
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)
        
        print(f"SAM3 loaded successfully on {self.device}")

    async def predict(
        self, 
        image: Image.Image, 
        boxes: Optional[List[List[float]]] = None,
        text_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate segmentation masks using SAM3.
        
        Args:
            image: PIL Image
            boxes: Optional bounding boxes (not used in current SAM3 API)
            text_prompt: Optional text prompt for segmentation
            
        Returns:
            Dictionary with masks, boxes, and scores
        """
        if self.model is None:
            await self.load()
        
        loop = asyncio.get_event_loop()
        
        def _inference():
            # Set image in processor
            inference_state = self.processor.set_image(image)
            
            if text_prompt:
                # Text-prompted segmentation
                output = self.processor.set_text_prompt(
                    state=inference_state, 
                    prompt=text_prompt
                )
                return output
            else:
                # For now, return empty result if no text prompt
                # SAM3's automatic mask generation requires different API
                return {"masks": [], "boxes": [], "scores": []}
        
        result = await loop.run_in_executor(None, _inference)
        
        # Handle different output formats
        if isinstance(result, dict):
            return {
                "masks": result.get("masks", []),
                "boxes": result.get("boxes", []),
                "scores": result.get("scores", [])
            }
        else:
            return {"masks": [], "boxes": [], "scores": []}
