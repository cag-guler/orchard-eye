import numpy as np
import cv2
from typing import Tuple, Dict

def run(self, image: WorkflowImageData) -> BlockResult:
    # Convert WorkflowImageData to numpy array
    img = image.numpy_image
    
    # If the image is not grayscale, convert to grayscale heatmap for processing
    if len(img.shape) == 3 and img.shape[2] > 1:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # Threshold the heatmap to create binary mask: you can customize threshold value
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours of flagged regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get coordinates of each flagged region (bounding rectangles)
    flagged_coords = []
    flagged_areas = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        flagged_coords.append((x, y, x+w, y+h))  # (min_x, min_y, max_x, max_y)
        area = cv2.contourArea(cnt)
        flagged_areas.append(area)

    # Sum of all flagged areas (in pixel units)
    total_flagged_area = np.sum(flagged_areas)

    # Get total area in pixels
    if len(gray.shape) == 2:
        total_area = gray.shape[0] * gray.shape[1]
    elif len(gray.shape) == 3:
        total_area = gray.shape[0] * gray.shape[1]
    else:
        total_area = 0  # fallback

    # Calculate ratio, avoid division by zero
    flagged_area_ratio = float(total_flagged_area) / float(total_area) if total_area > 0 else 0.0
    
    return {
        "flagged_coordinates": flagged_coords, # List of (min_x, min_y, max_x, max_y)
        "total_flagged_area": float(total_flagged_area),
        "flagged_area_ratio": float(flagged_area_ratio)
    }
