import cv2
from ultralytics import YOLO
import numpy as np

# Load a generic YOLOv8 model for prototyping
# In a real scenario with specific parts, you would load your custom trained model:
# model = YOLO('custom_lego_model.pt')
model = YOLO('yolov8n.pt')

# Define a mapping from generic object detection classes to dummy Lego part numbers for the prototype
# YOLOv8 default classes won't have Lego pieces, so we will pretend detected objects are Legos
# Or we can just use the generic YOLO model to find objects and assign them random Lego numbers for testing.

import random

# Common Lego part numbers as placeholders
LEGO_PART_NUMBERS = [
    "3001 (2x4 Brick)",
    "3003 (2x2 Brick)",
    "3020 (2x4 Plate)",
    "3022 (2x2 Plate)",
    "3710 (1x4 Plate)"
]

# Common Lego colors
LEGO_COLORS = ["Red", "Blue", "Yellow", "Black", "White", "Green", "Light Bluish Gray"]

def detect_lego_pieces(image_np):
    """
    Run the prototype detector on an image array.
    Returns:
        processed_image: Image with bounding boxes drawn.
        detections: A list of dictionaries containing info about each detected piece.
    """
    # Run YOLOv8 inference
    results = model(image_np)

    detections = []

    # Render the results on the image
    res = results[0]
    processed_image_bgr = res.plot() # Ultralytics function to draw bounding boxes, returns BGR
    # Convert BGR to RGB for Streamlit displaying
    processed_image = cv2.cvtColor(processed_image_bgr, cv2.COLOR_BGR2RGB)

    # Process the bounding boxes to create our simulated Lego list
    for box in res.boxes:
        # Get confidence score
        conf = float(box.conf[0])

        # We only care about somewhat confident detections
        if conf > 0.25:
            # Simulate a Lego part number and color based on the bounding box size/aspect ratio
            # (In the prototype, we just randomly assign them since YOLOv8 generic doesn't know Legos)

            # Use the box coordinates to seed a random choice so it's consistent for the same image
            coords = box.xyxy[0].tolist()
            seed_val = int(coords[0] + coords[1])
            random.seed(seed_val)

            part_name = random.choice(LEGO_PART_NUMBERS)
            color = random.choice(LEGO_COLORS)

            detections.append({
                "Part Number": part_name,
                "Color": color,
                "Confidence": f"{conf:.2f}",
                "Bounding Box": [round(c, 2) for c in coords]
            })

    return processed_image, detections
