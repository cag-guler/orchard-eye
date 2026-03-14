"""
Roboflow Workflow Python Block 2

Block Name: Merge Close Detections By Class
Block Description: Merges nearby 'apple' and 'damaged_apple' detections of the same class by IoU and distance threshold. Set iou_threshold and distance_threshold to control merging sensitivity.

Script: cluster_merge_block.py

This script contains the logic used inside a Roboflow Workflow
custom Python block for merging nearby detections into treatment zones.

The block merges detections of the same class (apple / damaged_apple)
based on:

- IoU overlap
- center distance threshold

This spatial merging step converts raw detections into
interpretable treatment regions used for orchard management.
"""

import numpy as np
import supervision as sv
from scipy.spatial.distance import cdist

def merge_distant_detections_by_class(detections: sv.Detections, iou_threshold: float = 0.0, distance_threshold: float = 1.0) -> sv.Detections:
    """
    Merges only those same-class detections that are actually overlapping (high IOU) or their centers are within a very small distance.
    For NO unintended merging, set iou_threshold = 0.0 and distance_threshold = 1.0 (effectively disables merging unless really overlapping/coincedent).
    """
    if len(detections) == 0:
        return detections

    target_classes = ["apple", "damaged_apple"]
    results = []

    for class_name in target_classes:
        # Filter detections for this class
        class_mask = np.array([
            detections.data["class_name"][i] == class_name
            for i in range(len(detections))
        ])
        class_detections = detections[class_mask]
        
        if len(class_detections) == 0:
            continue

        merged_mask = np.zeros(len(class_detections), dtype=bool)
        clusters = []
        centers = np.column_stack([
            (class_detections.xyxy[:, 0] + class_detections.xyxy[:, 2]) / 2,
            (class_detections.xyxy[:, 1] + class_detections.xyxy[:, 3]) / 2
        ])

        for i in range(len(class_detections)):
            if merged_mask[i]:
                continue
            cluster = [i]
            merged_mask[i] = True
            # IOU test
            for j in range(i + 1, len(class_detections)):
                if merged_mask[j]:
                    continue
                # compute IOU between box i and box j
                boxA = class_detections.xyxy[i]
                boxB = class_detections.xyxy[j]
                # intersection
                xx1 = max(boxA[0], boxB[0])
                yy1 = max(boxA[1], boxB[1])
                xx2 = min(boxA[2], boxB[2])
                yy2 = min(boxA[3], boxB[3])
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = inter / (areaA + areaB - inter + 1e-8)
                # center distance
                cxA, cyA = (boxA[0] + boxA[2]) / 2, (boxA[1] + boxA[3]) / 2
                cxB, cyB = (boxB[0] + boxB[2]) / 2, (boxB[1] + boxB[3]) / 2
                dist = np.hypot(cxA - cxB, cyA - cyB)
                if iou > iou_threshold or dist <= distance_threshold:
                    cluster.append(j)
                    merged_mask[j] = True
            clusters.append(cluster)

        merged_boxes = []
        confidences = []
        class_ids = []
        data = {k: [] for k in class_detections.data.keys()}
        for cluster in clusters:
            boxes = class_detections.xyxy[cluster]
            merged_box = np.array([
                np.min(boxes[:, 0]), np.min(boxes[:, 1]),
                np.max(boxes[:, 2]), np.max(boxes[:, 3])
            ])
            merged_boxes.append(merged_box)
            cluster_confidences = class_detections.confidence[cluster]
            confidences.append(np.max(cluster_confidences))
            class_ids.append(class_detections.class_id[cluster[0]])
            for k in data.keys():
                data[k].append(class_detections.data[k][cluster[0]])

        merged_class_detections = sv.Detections(
            xyxy=np.stack(merged_boxes) if merged_boxes else np.zeros((0, 4)),
            mask=None,
            confidence=np.array(confidences) if confidences else np.zeros((0,)),
            class_id=np.array(class_ids) if class_ids else np.zeros((0,)),
            tracker_id=None,
            data={k: np.array(v) for k, v in data.items()}
        )
        results.append(merged_class_detections)

    # Keep detections from other classes unchanged
    other_mask = np.array([
        detections.data["class_name"][i] not in target_classes
        for i in range(len(detections))
    ])
    other_detections = detections[other_mask]
    if len(other_detections) > 0:
        results.append(other_detections)

    if not results:
        return sv.Detections.empty()

    return sv.Detections.merge(results)


def run(self, detections: sv.Detections, iou_threshold: float = 0.0, distance_threshold: float = 1.0) -> dict:
    merged_detections = merge_distant_detections_by_class(detections, iou_threshold, distance_threshold)
    return {"merged_detections": merged_detections}
