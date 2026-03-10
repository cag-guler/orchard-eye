# orchard-eye

> Drone footage goes in. Treatment zones come out.

Rebuilding a computer vision agriculture pipeline using Roboflow Workflows to estimate targeted treatment zones in apple orchards.

**this is a demo.**

# Precision Orchard Vision

Fruit diseases spread locally. Most farms spray entire fields anyway.

This pipeline detects apple health from drone imagery and estimates which clusters actually need treatment — built with Roboflow Workflows, originally developed as my graduation project.

## Background

This project originated as my control and automation engineering graduation thesis: a precision agriculture system designed to detect crop health patterns from aerial imagery.

The original system included:

- YOLOv4 trained on custom datasets
- multi-objective crop grading based on disease, maturity, and water signals
- optimization algorithms (Genetic Algorithm + Big Bang–Big Crunch)
- MATLAB-based decision modeling
- MQTT infrastructure for field data communication

While the system worked, building the full pipeline required significant engineering overhead: dataset collection, labeling infrastructure, training pipelines, and large-image inference orchestration.

This repository rebuilds the vision layer using Roboflow Workflows to evaluate how modern developer tooling reduces that complexity.

So I rebuilt the core of it.

---

## The Problem

Traditional orchard management treats fields uniformly. But disease doesn't spread uniformly — it starts in clusters and expands outward. Spraying everything wastes resources and ignores the actual infection pattern.

A vision system that maps *where* the problem is changes the decision from "spray the field" to "spray these zones."

---

## Pipeline
```
Aerial Image
    → Image Slicing              (handle high-res drone footage)
    → Object Detection           (custom model trained on Roboflow)
    → Detection Stitching        (merge outputs across slices)
    → Cluster Analysis           (group nearby detections spatially)
    → Treatment Zone Map         (localized, actionable output)
```

---

## Model

Custom object detection model trained from scratch using Roboflow.

| Class | Description |
|-------|-------------|
| `apple` | Healthy fruit |
| `rotten_apple` | Diseased fruit |

> **Note:** Apple vs. rotten apple classification is used here as a fast proxy to validate the pipeline end-to-end.
> In real agricultural deployments, disease detection typically focuses on leaf-level pathology, where models learn from crop-specific morphology and disease patterns. These signals provide more actionable indicators for intervention decisions.  
> The current model intentionally simplifies the biological signal in order to test the vision pipeline itself before investing in more specialized dataset collection.

## Example Output

The system processes a high-resolution aerial image of an orchard and transforms raw visual data into spatial treatment signals.

Instead of simply detecting objects, the pipeline aggregates detections and analyzes their spatial proximity to estimate localized clusters of disease signals.

The output therefore moves beyond detection and produces a **decision-oriented map** that highlights areas of the orchard more likely to require intervention.

Pipeline output stages:

Drone Image  
→ Apple / Rotten Apple Detection  
→ Spatial Cluster Analysis  
→ Estimated Treatment Zones

The result is a simplified decision layer for orchard management — showing *where attention should be focused* rather than treating the entire field uniformly.



## Spatial Clustering Logic

Object detection alone does not directly answer the operational question:  
**where should treatment be applied?**

Individual detections only indicate isolated signals. In real agricultural environments, however, disease presence is meaningful when detections begin to appear **spatially grouped**.

To address this, the pipeline performs a spatial clustering step after detection stitching.

Bounding box coordinates from all detections are analyzed to identify groups of nearby detections that likely represent localized infection zones.

The clustering process follows three steps:

1. **Detection Stitching**  
   Predictions generated from sliced inference are merged back into the original image coordinate space.

2. **Spatial Proximity Analysis**  
   Bounding box centers are compared using Euclidean distance to determine spatial relationships between detections.

3. **Cluster Formation**  
   Nearby detections are grouped into clusters representing potential disease zones.

Instead of producing hundreds of independent detections, the system produces a smaller number of **interpretable treatment regions**.

This transforms the model output from a detection task into a **decision-support signal** that can guide localized spraying or inspection.


## What Roboflow Simplified

The original thesis pipeline required months of setup before a single 
inference could run — labeling infrastructure, training configuration, 
model versioning, large-image orchestration. Each layer had its own 
engineering overhead before the actual problem could be addressed.

With Roboflow, the model went from dataset to inference in hours.  
More importantly, the workflow builder allowed the full pipeline — 
slicing, detection, stitching, custom processing — to be assembled 
visually rather than wired together manually.

The result was a shift in where development time actually went: less 
on infrastructure, more on what the outputs mean and how they should 
be interpreted.

---

## Developer Observations

These are honest friction points encountered during development — not 
complaints, but the kind of feedback that comes from actually building 
on top of a platform.

**Python block constraints.**  
Custom Python blocks are the right mechanism for logic that doesn't 
fit the visual workflow — but the execution environment is restrictive. 
Limited dependencies, opaque errors, and publishing failures when 
runtime errors occur made iteration slower than expected. Better 
runtime feedback would meaningfully improve this.

**Slice stitching is a black box.**  
For large aerial images, sliced inference is necessary — but reasoning 
about how detections from adjacent slices are reconciled is difficult. 
When detections conflicted at slice boundaries, there was no clear way 
to inspect or adjust the stitching behavior. Intermediate output 
visibility at this stage would help significantly.

**Grid-based output isn't native.**  
The natural output format for precision agriculture isn't clustered 
polygons — it's a field grid with per-cell recommendations. Getting 
from detection clusters to actionable grid coordinates required custom 
logic entirely outside the Workflow environment. For agriculture 
deployments specifically, this is a meaningful gap.

---

## Conclusion

The thesis system worked. Building it revealed how much engineering 
effort goes into problems that aren't the core problem.

Roboflow removes most of that overhead. Dataset preparation, model 
training, and basic pipeline orchestration are no longer the hard 
parts. The hard parts — spatial reasoning, decision logic, bridging 
model output to field action — still require custom work. That 
boundary is now much further along the pipeline than it used to be.

That's a meaningful shift.
