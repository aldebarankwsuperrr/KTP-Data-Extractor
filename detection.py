import numpy as np
from math import dist

def generate_important_points(image, segment_model, obb_model):
    try:
        # Run object detection and segmentation
        segment_results = segment_model(image)
        obb_results = obb_model(image)

        # Extract oriented bounding box points and segment boxes/masks
        obb_points = np.int64(obb_results[0].obb.xyxyxyxy[0])  # Extracting OBB points
        segment_boxes = np.int64(segment_results[0].boxes.xyxy[0])  # Extracting segment boxes
        segment_masks = np.int64(segment_results[0].masks.xy[0])  # Extracting segment masks
    
        return obb_points, segment_boxes, segment_masks
    except IndexError as e:
        print(f"Error: {e}. There are no detected objects in the image.")
        return None, None, None

def synchronize_point(bbox_point, obb_points):
    distance = dist(bbox_point, obb_points[0])
    shortest = obb_points[0]
    for i in range(len(obb_points)):
        new_distance = dist(obb_points[i], bbox_point)  
        if distance > new_distance:
            distance = new_distance
            shortest = obb_points[i]
    return shortest

def shortest_point(point, mask):
    distances = [dist(point, m) for m in mask]
    return mask[np.argmin(distances)]

def generate_obb_corners(segment_boxes, obb_points):
    obb_top_left = synchronize_point(segment_boxes[:2], obb_points)
    obb_top_right = synchronize_point([segment_boxes[2], segment_boxes[1]], obb_points)
    obb_bottom_left = synchronize_point([segment_boxes[0], segment_boxes[3]], obb_points)
    obb_bottom_right = synchronize_point([segment_boxes[2], segment_boxes[3]], obb_points)
    
    return obb_top_left, obb_top_right, obb_bottom_left, obb_bottom_right

def generate_mask_corners(obb_top_left, obb_top_right, obb_bottom_left, obb_bottom_right, segment_masks):
    mask_top_left = shortest_point(obb_top_left, segment_masks)
    mask_top_right = shortest_point(obb_top_right, segment_masks)
    mask_bottom_left = shortest_point(obb_bottom_left, segment_masks)
    mask_bottom_right = shortest_point(obb_bottom_right, segment_masks)
    
    return mask_top_left, mask_top_right, mask_bottom_left, mask_bottom_right
