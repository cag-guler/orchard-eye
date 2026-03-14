
# orchard-eye

> **Drone imagery → damaged apple detection → spatial treatment zones**

A computer vision pipeline that converts **drone imagery of apple orchards into localized treatment zones**.

Instead of simply detecting fruit, the system analyzes **spatial clusters of damaged apples** to estimate where intervention is actually needed.

This project rebuilds part of my **Control and Automation Engineering graduation thesis** using **Roboflow Workflows** to explore how modern tooling simplifies real-world computer vision pipelines.

---

# 🚀 Quick Overview

The pipeline converts raw drone imagery into **decision-oriented signals for orchard management**.

```
Drone Image
     ↓
Object Detection (apple / damaged_apple)
     ↓
Detection Stitching
     ↓
Spatial Clustering
     ↓
Treatment Zone Estimation
```

### Example Pipeline Result

| Input Image | Detection | Treatment Zones |
|-------------|-----------|-----------------|
| ![](assets/images/field.png) | ![](assets/images/detection.jpeg) | ![](assets/images/damaged_zone120.jpeg) |

The final output highlights **areas where disease signals cluster**, enabling **targeted spraying instead of uniform field treatment**.

---

# 🌱 The Problem

Fruit diseases rarely spread uniformly across orchards.

They usually begin in **small localized clusters** before spreading outward.

However, many orchards still rely on **uniform spraying**, treating entire fields regardless of where infection actually exists.

This approach:

- wastes chemicals
- increases operational costs
- ignores spatial disease patterns

A vision system that maps **where disease signals concentrate** changes the decision from:

```
spray the entire field
```

to

```
spray these zones
```

---

# 🤖 Model

Custom object detection model trained using **Roboflow**.

| Class | Description |
|------|-------------|
| **apple** | healthy fruit |
| **damaged_apple** | diseased fruit |

Apple vs damaged apple classification is used as a **proxy signal** to validate the spatial reasoning pipeline.

In real agricultural deployments, disease detection often focuses on **leaf pathology and plant morphology**, but fruit-level signals allow rapid validation of the full vision system.

---

# 🔎 Detection Output

The model detects apples and damaged apples across aerial orchard imagery.

![Detection Output](assets/images/detection.jpeg)

These detections form the **base signals used for clustering and treatment zone estimation**.

---

# 📍 Spatial Cluster Analysis

Object detection alone does **not answer the operational question:**

> **Where should treatment be applied?**

Individual detections represent isolated signals. In agricultural environments, disease presence becomes meaningful when detections appear **spatially grouped**.

The pipeline therefore performs **spatial clustering on bounding box coordinates**.

### Clustering Process

1️⃣ **Detection Stitching**  
Predictions from sliced inference are mapped back into the original image coordinate space.

2️⃣ **Spatial Proximity Analysis**  
Bounding box centers are compared using **Euclidean distance**.

3️⃣ **Cluster Formation**  
Nearby detections are merged into **treatment zones**.

---

# 📊 Merge Threshold Comparison

Different clustering thresholds were evaluated to understand how spatial grouping affects treatment zone estimation.

### 80px merge

![](assets/images/zone80.jpeg)

### 120px merge

![](assets/images/zone120.jpeg)

### 180px merge

![](assets/images/zone180.jpeg)

### 240px merge

![](assets/images/zone240.jpeg)

---

# ✅ Optimal Treatment Zones

A merge threshold of **120 pixels** produced the most interpretable treatment regions.

This threshold balances two competing effects:

- small thresholds fragment clusters into many tiny zones  
- large thresholds merge distant detections into overly large regions  

The **120px merge distance** produced the clearest spatial grouping of damaged apples.

![Optimal Zones](assets/images/damaged_zone120.jpeg)

These zones represent areas **most likely to require intervention**.

---

# 🔥 Disease Density Heatmap

In addition to clustering, the pipeline generates a **spatial heatmap** showing where damaged apple detections accumulate.

![Heatmap](assets/images/heatmap-rotten.jpeg)

This visualization highlights **disease intensity across the orchard**.

---

# 📈 Quantitative Signals

Beyond visual outputs, the workflow extracts simple quantitative indicators describing orchard health.

### 1️⃣ Infection Ratio

```
damaged_apple / total_apple
```

Represents the **proportion of detected fruit that appears damaged**.

---

### 2️⃣ Disease Density Coverage

```
damaged_heatmap_area / total_field_area
```

Measures how much of the orchard contains **measurable disease density**.

---

### 3️⃣ Treatment Zone Coverage

```
merged_damaged_zone_area / total_field_area
```

Using the **120px merge threshold**, this metric estimates how much of the orchard **may require targeted intervention**.

Together these signals transform raw detections into **decision-oriented indicators for orchard management**.

---

# ⚙️ Roboflow Workflow Architecture

![Workflow](assets/images/workflow-diagram.png)

Key stages include:

- image slicing for large aerial imagery  
- object detection inference  
- detection stitching  
- spatial clustering via custom Python blocks  
- heatmap generation  
- quantitative signal extraction  

The workflow produces both **visual outputs** and **structured signals** describing orchard conditions.

---

# 🎓 Background

This project originated as my **Control and Automation Engineering graduation thesis** focused on **precision agriculture systems**.

The original research pipeline included:

- **YOLOv4 models** trained on agricultural datasets  
- **multi-objective crop grading** based on disease and maturity signals  
- **optimization algorithms** (Genetic Algorithm + Big Bang–Big Crunch)  
- **MATLAB-based decision modeling**  
- **MQTT infrastructure** for transmitting field data  

While the system worked, building the full pipeline required significant engineering overhead.

Most development time was spent on:

- dataset collection and labeling
- training pipeline configuration
- large-image inference orchestration
- model infrastructure

Rather than focusing on the **actual decision logic**.

This repository rebuilds the **vision layer using Roboflow Workflows** to evaluate how modern tooling reduces that overhead and allows development effort to focus on:

> **turning model outputs into actionable agricultural decisions**
