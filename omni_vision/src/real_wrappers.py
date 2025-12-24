import asyncio
import os
import torch
import requests
import base64
import io
import cv2
from typing import List, Any
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
from .config import Config, ModelType
from .model_wrappers import ModelWrapper, DetectionResult
from transformers import PreTrainedModel
# Monkey Patch for Florence-2 compatibility with newer/older transformers
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False


class YOLOv12RealWrapper(ModelWrapper):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    async def load(self):
        print(f"Loading YOLOv12 from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            print("YOLOv12 loaded.")
        except Exception as e:
            print(f"Failed to load YOLOv12: {e}")

    async def predict(self, image: Any, **kwargs) -> List[DetectionResult]:
        if not self.model: return []
        # Ultralytics handles paths, PIL, numpy
        results = await asyncio.to_thread(self.model.predict, image, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(DetectionResult(
                    label=self.model.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=box.xyxy[0].tolist()
                ))
        return detections

class RFDETRRealWrapper(ModelWrapper):
    """
    RF-DETR: Accessed via Roboflow API directly (Requests).
    """
    def __init__(self, model_id: str = "rf-detr-v1"):
        self.model_id = model_id
        self.api_key = Config.ROBOFLOW_API_KEY

    async def load(self):
        if not self.api_key:
            print("Warning: ROBOFLOW_API_KEY missing.")
        else:
            print("RF-DETR (API) initialized.")

    async def predict(self, image: Any, **kwargs) -> List[DetectionResult]:
        if not self.api_key: return []
        
        # Prepare Image for API (Base64)
        img_str = ""
        if isinstance(image, str) and os.path.exists(image):
            with open(image, "rb") as f:
                img_data = f.read()
                img_str = base64.b64encode(img_data).decode("utf-8")
        elif isinstance(image, bytes):
            img_str = base64.b64encode(image).decode("utf-8")
        elif hasattr(image, "tolist"): # Numpy (OpenCV)
            _, buf = cv2.imencode(".jpg", image)
            img_str = base64.b64encode(buf).decode("utf-8")
        
        if not img_str: return []

        # Construct URL
        url = f"https://detect.roboflow.com/{self.model_id}?api_key={self.api_key}"
        
        try:
            resp = await asyncio.to_thread(
                requests.post, 
                url, 
                data=img_str, 
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if resp.status_code != 200:
                print(f"RF-DETR API Error: {resp.text}")
                return []
            
            data = resp.json()
            detections = []
            for p in data.get('predictions', []):
                # Roboflow returns x,y (center)
                x, y, w, h = p['x'], p['y'], p['width'], p['height']
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                
                detections.append(DetectionResult(
                    label=p['class'],
                    confidence=p['confidence'],
                    box=[x1, y1, x2, y2]
                ))
            return detections
        except Exception as e:
            print(f"RF-DETR Request Failed: {e}")
            return []

class Florence2Wrapper(ModelWrapper):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.last_query = ""

    async def load(self):
        print(f"Loading Florence-2 from {self.model_path}...")
        try:
            self.model = await asyncio.to_thread(
                AutoModelForCausalLM.from_pretrained, 
                self.model_path, 
                trust_remote_code=True
            )
            self.processor = await asyncio.to_thread(
                AutoProcessor.from_pretrained, 
                self.model_path, 
                trust_remote_code=True
            )
            print("Florence-2 loaded.")
        except Exception as e:
             print(f"Error loading Florence-2: {e}")

    async def predict(self, image: Any, text_query: str = "", **kwargs) -> List[DetectionResult]:
        if not self.model or not text_query: return []
        
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        # If the model expects specific formatting for queries, adjust here.
        # Florence-2 OVD usually takes Task + Text
        prompt = task_prompt + text_query
        
        # Determine image format
        image_obj = None
        if isinstance(image, str):
            image_obj = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
             image_obj = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image,  (torch.Tensor, list, tuple)):
             # Handle other formats if strictly needed
             pass
        else: # Numpy/CV2 likely
             image_obj = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not image_obj: return []

        inputs = self.processor(text=prompt, images=image_obj, return_tensors="pt")
        
        generated_ids = await asyncio.to_thread(
            self.model.generate,
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=(image_obj.width, image_obj.height))
        
        data = parsed_answer.get(task_prompt, {})
        bboxes = data.get('bboxes', [])
        labels = data.get('bboxes_labels', [])
        
        detections = []
        for box, label in zip(bboxes, labels):
            detections.append(DetectionResult(
                label=label,
                confidence=0.99, 
                box=box
            ))
        return detections

# --- SAM3 Utilities (from notebook) ---
from typing import List
import torch
import numpy as np
from PIL import Image as PILImage
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3 import build_sam3_image_model
import sam3
sam3_root = os.path.dirname(sam3.__file__)
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device

# Global counter for SAM3 IDs
GLOBAL_SAM3_COUNTER = 1

def create_empty_datapoint():
    return Datapoint(find_queries=[], images=[])

def set_image(datapoint, pil_image):
    w,h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h,w])]

def add_visual_prompt(datapoint, boxes: List[List[float]], labels: List[bool], text_prompt="visual"):
    global GLOBAL_SAM3_COUNTER
    # boxes expected in XYXY
    if not boxes: return
    labels_tensor = torch.tensor(labels, dtype=torch.bool).view(-1)
    w, h = datapoint.images[0].size
    
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_prompt,
            image_id=0,
            object_ids_output=[], 
            is_exhaustive=True,
            query_processing_order=0,
            input_bbox=torch.tensor(boxes, dtype=torch.float).view(-1,4),
            input_bbox_label=labels_tensor,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_SAM3_COUNTER,
                original_image_id=GLOBAL_SAM3_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            )
        )
    )
    QUERY_ID = GLOBAL_SAM3_COUNTER - 1 # Use current counter as ID (0-indexed logic in notebook is slightly weird, adapting)
    GLOBAL_SAM3_COUNTER += 1 # Increment for next
    return QUERY_ID

class SAM3RealWrapper(ModelWrapper):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cpu" # Force CPU on Mac

    async def load(self):
        print(f"Loading SAM3 (Standard) from {self.model_path}... Device: {self.device}")
        try:
            # Import here to avoid early failures
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            bpe_path = os.path.join(sam3_root, "assets/bpe_simple_vocab_16e6.txt.gz")
            
            # Check if model path exists
            if os.path.exists(self.model_path):
                 self.model = build_sam3_image_model(
                    bpe_path=bpe_path, 
                    checkpoint_path=self.model_path, 
                    device=self.device
                )
            else:
                print(f"Warning: SAM3 weights not found at {self.model_path}. Skipping load to avoid hanging on gated download.")
                self.model = None
                return
            
            # Init Processor
            self.processor = Sam3Processor(self.model)
            # Ensure processor device is CPU? It's set in __init__ usually.
            # If patched, it should be fine.  
            
            print("SAM3 (Standard) loaded successfully.")
        except Exception as e:
            print(f"Failed to load SAM3 (Standard): {e}")
            self.model = None

    async def predict(self, image: Any, boxes: List[List[float]] = None, text_query: str = None, **kwargs) -> List[Any]:
        if not self.model or not self.processor: return []
        if not boxes and not text_query: return []
        
        try:
            # 1. Prepare Image
            if isinstance(image, str):
                pil_image = PILImage.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image # Assume PIL
            
            # 2. Set Image
            # Note: processor methods are synchronous torch code.
            # We wrap the whole block or key parts in thread if blocking is an issue.
            # For now, let's just run it.
            
            state = self.processor.set_image(pil_image)
            
            # 3. Add Prompts
            # If we have boxes from YOLO, use them.
            if boxes:
                # SAM3 add_geometric_prompt takes one box at a time? 
                # Doc says: "box: List ... [center_x, center_y, width, height] and normalized in [0, 1]"
                # Wait, doc says: "The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range."
                # YOLO boxes are [x1, y1, x2, y2] in pixels. We need to convert.
                
                w, h = pil_image.size
                
                # We need to accumulate results? Or can we pass multiple boxes?
                # add_geometric_prompt(box: List, label: bool, state: Dict) seems single box?
                # "adding a batch and sequence dimension ... boxes = ... .view(1, 1, 4)"
                # It appends to state["geometric_prompt"].
                # So we can loop through boxes.
                
                for box_xyxy in boxes:
                    x1, y1, x2, y2 = box_xyxy
                    # Convert to cx, cy, w, h normalized
                    bw = x2 - x1
                    bh = y2 - y1
                    cx = x1 + bw / 2
                    cy = y1 + bh / 2
                    
                    norm_box = [cx / w, cy / h, bw / w, bh / h]
                    
                    # Add prompt (positive label)
                    state = self.processor.add_geometric_prompt(box=norm_box, label=True, state=state)
            
            if text_query:
                # If text query provided, use it.
                state = self.processor.set_text_prompt(prompt=text_query, state=state)

            # 4. Results
            # State now contains results: state["masks"], state["boxes"], state["scores"]
            # Masks: [N, H, W] boolean?
            
            masks_out = []
            if "masks" in state:
                # state["masks"] is tensor
                masks_tensor = state["masks"]
                if hasattr(masks_tensor, "cpu"):
                    masks_numpy = masks_tensor.cpu().numpy() # [N, H, W] or [N, 1, H, W]?
                    # Check shape
                    if masks_numpy.ndim == 4:
                        masks_numpy = masks_numpy.squeeze(1) # Remove channel if present
                    
                    # Convert to list of masks
                    masks_out = [m for m in masks_numpy]

            # Clear state for next run?
            # self.processor.reset_all_prompts(state) # If we reused state object. Here we create new one from set_image.
            
            return masks_out

        except Exception as e:
            print(f"SAM3 Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return []

