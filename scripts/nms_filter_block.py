"""
Roboflow Workflow Python Block 3

Block Name: Non Maximum Suppression
Block Description: Performs non-maximum suppression to filter overlapping detections using an IoU threshold. Outputs filtered detections.

Script: nms_filter_block.py

Performs Non-Maximum Suppression (NMS) on detection results
to remove overlapping bounding boxes.

This step ensures that duplicate detections are filtered
before spatial clustering and zone generation.

Inputs:
    detections (sv.Detections)

Outputs:
    filtered detections after NMS
"""

import numpy as np
import supervision as sv

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return interArea / (areaA + areaB - interArea + 1e-6)


def run(self, detections: sv.Detections, iou_threshold: float = 0.35):
    if len(detections) == 0:
        return {"detections": detections}

    boxes = detections.xyxy
    conf = detections.confidence

    keep = np.ones(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i+1, len(boxes)):
            if not keep[j]:
                continue
            iou = compute_iou(boxes[i], boxes[j])
            if iou > iou_threshold:
                if conf[i] >= conf[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    filtered = detections[keep]
    return {"detections": filtered}
