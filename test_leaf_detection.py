#!/usr/bin/env python3
"""Test script to verify leaf detection is working properly"""

import numpy as np
import cv2
from PIL import Image
import os


def detect_leaf_in_image(image_array) -> dict:
    """Test version of leaf detection function"""

    # Denormalize for OpenCV processing
    img_uint8 = (image_array * 255).astype(np.uint8)

    # METHOD 1: GREEN COLOR DETECTION using HSV
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # Green hue range in HSV (more strict)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.sum(green_mask > 0)
    total_pixels = green_mask.shape[0] * green_mask.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100

    # METHOD 2: TEXTURE ANALYSIS
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = np.std(laplacian)

    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_percentage = (np.sum(edges > 0) / total_pixels) * 100

    # METHOD 3: COLOR VARIANCE
    color_variance = np.std(img_uint8)

    # METHOD 4: MORPHOLOGICAL ANALYSIS
    contours, _ = cv2.findContours(
        green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour_area = 0
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        largest_contour_area = max(areas) if areas else 0

    largest_contour_percentage = (largest_contour_area / total_pixels) * 100

    # VALIDATION CRITERIA
    has_green = green_percentage >= 20
    has_edges = edge_percentage > 8
    has_variance = color_variance > 30
    has_coherent_green = largest_contour_percentage > 5
    has_texture = texture_score > 15

    is_leaf = (
        has_green and has_edges and has_variance and has_coherent_green and has_texture
    )

    return {
        "is_leaf": is_leaf,
        "green_percentage": green_percentage,
        "edge_percentage": edge_percentage,
        "texture_score": texture_score,
        "color_variance": color_variance,
        "largest_contour_percentage": largest_contour_percentage,
        "criteria": {
            "has_green": has_green,
            "has_edges": has_edges,
            "has_variance": has_variance,
            "has_coherent_green": has_coherent_green,
            "has_texture": has_texture,
        },
    }


# Test with a sample image
test_images_dir = "test_images"
if os.path.exists(test_images_dir):
    print(f"Testing images in {test_images_dir}...")
    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(test_images_dir, filename)
            try:
                # Load and prepare image
                img = Image.open(filepath)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize to 224x224
                img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized).astype(np.float32) / 255.0

                # Test detection
                result = detect_leaf_in_image(img_array)

                print(f"\n{'='*60}")
                print(f"Image: {filename}")
                print(f"{'='*60}")
                print(f"Is Leaf: {result['is_leaf']}")
                print(f"\nMetrics:")
                print(
                    f"  Green Percentage: {result['green_percentage']:.2f}% (need ≥20%)"
                )
                print(f"  Edge Percentage: {result['edge_percentage']:.2f}% (need >8%)")
                print(f"  Texture Score: {result['texture_score']:.2f} (need >15)")
                print(f"  Color Variance: {result['color_variance']:.2f} (need >30)")
                print(
                    f"  Largest Region: {result['largest_contour_percentage']:.2f}% (need >5%)"
                )

                print(f"\nCriteria Met:")
                for criterion, met in result["criteria"].items():
                    status = "✓ PASS" if met else "✗ FAIL"
                    print(f"  {criterion}: {status}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
else:
    print(f"Directory '{test_images_dir}' not found. Create it and add test images.")
