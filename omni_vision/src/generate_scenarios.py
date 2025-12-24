import cv2
import numpy as np

def create_image(filename, width, height, color):
    # Create a solid color image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    
    # Add info text
    text = f"{filename} ({width}x{height})"
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(filename, img)
    print(f"Generated {filename}")

if __name__ == "__main__":
    # 1. Normal Scenario (Small size -> Normal density)
    create_image("scenario_normal.jpg", 640, 480, (200, 100, 0)) # Blue-ish

    # 2. Dense Scenario (Large size -> High density > 15 objects)
    # Our updated mock checks for width > 800
    create_image("scenario_dense.jpg", 1280, 720, (0, 0, 200)) # Red-ish

    # 3. Intent Scenario (Normal size, intent provided via CLI)
    create_image("scenario_intent.jpg", 640, 480, (0, 100, 0)) # Green-ish
