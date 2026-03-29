#!/usr/bin/env python3
"""Test script to verify STRICTER leaf detection is working"""

import numpy as np
import cv2
from PIL import Image
import os


def detect_leaf_in_image_strict(image_array) -> dict:
    """Stricter leaf detection function"""

    img_uint8 = (image_array * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)

    # STRICT green range
    lower_green = np.array([35, 50, 40])
    upper_green = np.array([80, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = np.sum(green_mask > 0)
    total_pixels = green_mask.shape[0] * green_mask.shape[1]
    green_percentage = (green_pixels / total_pixels) * 100

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_score = np.std(laplacian)

    edges = cv2.Canny(gray, 50, 150)
    edge_percentage = (np.sum(edges > 0) / total_pixels) * 100

    color_variance = np.std(img_uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    green_closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        green_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour_area = 0
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        largest_contour_area = max(areas) if areas else 0

    largest_contour_percentage = (largest_contour_area / total_pixels) * 100

    avg_saturation = np.mean(hsv[:, :, 1])
    saturation_ok = 40 < avg_saturation < 240

    has_green = green_percentage >= 25
    has_edges = edge_percentage > 10
    has_variance = color_variance > 35
    has_coherent_green = (
        largest_contour_percentage > 5 and largest_contour_percentage < 80
    )
    has_texture = texture_score > 20
    has_saturation = saturation_ok

    is_leaf = (
        has_green
        and has_edges
        and has_variance
        and has_coherent_green
        and has_texture
        and has_saturation
    )

    return {
        "is_leaf": is_leaf,
        "green_pct": green_percentage,
        "edge_pct": edge_percentage,
        "texture": texture_score,
        "variance": color_variance,
        "region_pct": largest_contour_percentage,
        "saturation": avg_saturation,
        "criteria": {
            "has_green (>=25%)": has_green,
            "has_edges (>10%)": has_edges,
            "has_variance (>35)": has_variance,
            "has_coherent": has_coherent_green,
            "has_texture (>20)": has_texture,
            "saturation_ok": has_saturation,
        },
    }


test_images_dir = "test_images"
if os.path.exists(test_images_dir):
    print("Testing STRICTER Leaf Detection\n")
    for filename in os.listdir(test_images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(test_images_dir, filename)
            try:
                img = Image.open(filepath)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized).astype(np.float32) / 255.0

                result = detect_leaf_in_image_strict(img_array)

                print(f"{'='*70}")
                print(f"File: {filename}")
                print(
                    f"RESULT: {'✓ VALID LEAF' if result['is_leaf'] else '✗ NOT A LEAF'}"
                )
                print(f"{'='*70}")
                print(
                    f"Green: {result['green_pct']:.1f}% | Edge: {result['edge_pct']:.1f}% | Texture: {result['texture']:.1f}"
                )
                print(
                    f"Variance: {result['variance']:.1f} | Region: {result['region_pct']:.1f}% | Saturation: {result['saturation']:.0f}"
                )

                for criterion, met in result["criteria"].items():
                    status = "✓" if met else "✗"
                    print(f"  {status} {criterion}")
                print()

            except Exception as e:
                print(f"ERROR {filename}: {e}\n")
else:
    print(f"No test_images directory found")
