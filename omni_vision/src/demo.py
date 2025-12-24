import asyncio
import cv2
import argparse
import numpy as np
import sys
import os
from src.pipeline import OmniVisionPipeline
from src.config import ModelType

def draw_detections(frame, detections, meta):
    """Draws bounding boxes and labels on the frame."""
    frame_h, frame_w = frame.shape[:2]
    
    # Draw processing mode
    cv2.putText(frame, f"Mode: {meta['processing_mode']}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for det in detections:
        box = det['box'] # [x1, y1, x2, y2]
        label = det['label']
        conf = det['confidence']
        
        # Ensure box is integer
        x1, y1, x2, y2 = map(int, box)
        
        # Draw Rectangle
        color = (0, 255, 255) # Yellow
        if "RF-DETR" in meta['processing_mode']:
            color = (0, 0, 255) # Red for high precision
        elif "DINO-X" in meta['processing_mode']:
             color = (255, 0, 0) # Blue for intent
             
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label
        label_text = f"{label} {conf:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

async def process_image(pipeline, image_path, text_query=None):
    print(f"Processing image: {image_path}...")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    # Pipeline expects bytes usually for web, but our mock accepts Any.
    # However, to simulate 'reading', we can just pass the frame.
    # Logic note: real pipeline might need bytes or PIL. 
    # Current mock implementation: predict(image) -> sleeps -> returns mock.
    # So passing numpy array `frame` is fine.
    
    result = await pipeline.analyze(image=frame, text_query=text_query)
    
    # Visualize
    output = draw_detections(frame, result['detections'], result['meta'])
    
    output_path = "output_demo.jpg"
    cv2.imwrite(output_path, output)
    print(f"Saved result to {output_path}")

async def process_video(pipeline, video_path, text_query=None):
    print(f"Processing video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    output_path = "output_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Process every frame? Or skip for speed in demo? 
        # Let's process every 5th frame to simulate real-time on CPU mock or just every frame.
        # Mock latency is high (0.05s - 0.2s), so video generation will be slow.
        
        print(f"Processing frame {frame_count}...", end='\r')
        result = await pipeline.analyze(image=frame, text_query=text_query)
        output_frame = draw_detections(frame, result['detections'], result['meta'])
        out.write(output_frame)

    cap.release()
    out.release()
    print(f"\nSaved video result to {output_path}")

async def main():
    parser = argparse.ArgumentParser(description="Omni-Vision Verification Demo")
    parser.add_argument("input", help="Path to input image or video")
    parser.add_argument("--query", help="Text query for DINO-X", default=None)
    
    args = parser.parse_args()
    
    # Initialize Pipeline
    pipeline = OmniVisionPipeline()
    print("Loading models...")
    await pipeline.load_models()
    
    # Detect File Type
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        await process_image(pipeline, args.input, args.query)
    elif ext in ['.mp4', '.avi', '.mov']:
        await process_video(pipeline, args.input, args.query)
    else:
        print("Unsupported file format. Please use .jpg, .png, .mp4, etc.")

if __name__ == "__main__":
    asyncio.run(main())
