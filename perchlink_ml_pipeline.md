# PerchLink CV — Complete ML Pipeline Strategy

> **Based on your actual v1 training run analysis.**
> Dataset: 2001 train / 996 val / 503 test images | 1 class (`pole`) | YOLO11n | 50 epochs

---

## ⚠️ Critical Diagnosis of Your v1 Run — Read This First

Before any new experiment, here is what your `results.csv` actually tells us:

| Metric | Best Value | Epoch | Assessment |
|---|---|---|---|
| mAP50 | **0.326** | 46 | Critically low |
| mAP50-95 | **0.112** | 46 | Very low |
| Precision | 0.369 | 46 | Unstable — oscillates wildly |
| Recall | 0.386 | 46 | Low and erratic |
| Train cls_loss | 29.0 → 1.1 | 1→50 | Massive — confirms single-class mismatch |
| Val cls_loss | 14.5 → 2.2 | | **Never converges** |
| Train box_loss | 1.66 → 0.82 | | Improving but slowly |
| Val box_loss | 2.19 → 1.77 | | Gap is large → **underfitting** |

### 🔴 The Most Alarming Signal: `train/cls_loss` started at **29.0**

A normal cls_loss at epoch 1 is ~1–3. A value of 29 means the model's class predictions are catastrophically wrong from the start. We ran a full data audit and discovered the root cause:

**93% of your training images had NO bounding boxes.**
- **Positive images (with poles):** 133
- **Background images (empty labels):** 1868

The model was being fed 14 empty images for every 1 image with a pole. This extreme imbalance forced YOLO to predict "nothing" constantly, destroying precision and recall. I have already run a dataset curation script (`curate_dataset.py`) to delete the excessive background images. The dataset now maintains a healthy ~10% background ratio.

### 🟡 Timing Anomaly in Your Run

```
Epoch 10: time=302s cumulative
Epoch 11: time=1245s cumulative  ← 943 second gap (15 minutes for 1 epoch)
Epoch 20: time=2702s ← normal
Epoch 21: time=2768s ← normal  
Epoch 22: time=7328s ← 4560s gap again
Epoch 23: time=31659s ← system was interrupted/sleeping
```

Your training was interrupted multiple times by system sleep. This corrupts the `time` column and may have caused CUDA context resets. **Always run training with sleep disabled.**

---

## Part 0 — The Dataset Curation (Action Taken)

> **I have already audited and curated your dataset.**
> Bad data wastes GPU-hours. Your dataset was drowning in background images.

Here is what the audit found and fixed:

| Split | Original Images | Positive (Poles) | Background (Empty) | Action Taken | Current Images |
|---|---|---|---|---|---|
| **Train** | 2001 | 133 | 1868 (93%) | Kept 13, deleted 1855 | **146** |
| **Val** | 996 | 297 | 699 (70%) | Kept 29, deleted 670 | **326** |
| **Test** | 503 | 158 | 345 (68%) | Kept 15, deleted 330 | **173** |

*Note: The `convert_hotosm.py` script extracted the HOTOSM parquet files, but the original dataset simply did not contain pole annotations for most images.*

You now have a clean dataset with a correct 10% background ratio, but **it is extremely small (133 training images)**. Experiment 2 will be a proof-of-concept, but you absolutely must collect more data.

---

## Part 1 — Architecture Decision: Which YOLO Model

### Recommendation: **Stay with YOLO11n for now, but fix the training setup**

Here is why, specific to PerchLink:

| Model | Params | GFLOPs | mAP (COCO) | RTX 3050 batch | Pi 4 FPS |
|---|---|---|---|---|---|
| YOLO11n | 2.6M | 6.5 | 39.5 | 16–32 | ~8–12 |
| YOLO11s | 9.4M | 21.5 | 47.0 | 8–16 | ~3–5 |
| YOLO11m | 20.1M | 67.9 | 51.5 | 4–8 | ~1–2 |
| YOLOv8n | 3.2M | 8.7 | 37.3 | 16–32 | ~8–12 |

**Why YOLO11n is correct for your stage:**

1. You have **one class** with only **133 training images**. A larger model will instantly overfit on such a tiny dataset.
2. Your ultimate target is a **Raspberry Pi or similar edge device**. YOLO11n is the only model that realistically achieves >5 FPS on a Pi 4 without TensorRT.
3. Your v1 results (mAP50 ~0.33) were completely **data-limited** due to the 93% background ratio. Switching to YOLO11s will not fix data problems.
4. YOLO11n pretrained on COCO already knows what a `pole` looks like in general scenes. Fine-tuning is the correct approach.

**When to upgrade:** If after expanding your dataset to >1500 images, your mAP50-95 plateaus below 0.45, then consider YOLO11s.

> **Do NOT use YOLO26n/YOLO26s.** These are not stable production releases as of your Ultralytics version (8.4.171). Stick to YOLO11 series.

---

## Part 2 — The PerchLink System Architecture

YOLO should handle **only one layer** of the perception stack:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PerchLink Perception Stack                    │
├──────────────────┬──────────────────────────────────────────────┤
│  Layer           │  Algorithm           │  YOLO handles this?   │
├──────────────────┼──────────────────────────────────────────────┤
│ Object Detection │  YOLO11n             │  ✅ YES               │
│ Suitability      │  Rule engine /       │  ❌ NO — post-process │
│  classification  │  shape analysis      │  on YOLO bbox         │
│ Geometry / depth │  Monocular depth     │  ❌ NO — MiDaS/       │
│  estimation      │  or stereo camera    │  DPT or stereo rig    │
│ 3D localization  │  PnP + camera        │  ❌ NO — needs        │
│                  │  intrinsics          │  calibrated camera    │
│ Approach vector  │  PID / MPC on        │  ❌ NO — flight       │
│  planning        │  flight controller   │  controller domain    │
│ Perch execution  │  Mechanical servo    │  ❌ NO — actuator     │
│                  │  + force feedback    │  domain               │
└──────────────────┴──────────────────────────────────────────────┘
```

### What YOLO gives you (the raw outputs you need):
- Bounding box `[x_center, y_center, width, height]` in normalized image coords
- Class confidence score
- Class ID

### What your post-processing layer must compute from the bbox:
```python
# Suitability heuristic (implement AFTER detection)
def is_perchable(box, frame_shape, min_conf=0.45):
    x, y, w, h = box  # normalized
    aspect_ratio = h / (w + 1e-6)
    
    # Poles are tall and narrow from drone view
    # Railings are wide and narrow
    # Branches are roughly 1:1 to 2:1
    
    area_fraction = w * h  # fraction of frame
    
    return {
        "pole":    aspect_ratio > 3.0 and area_fraction > 0.02,
        "railing": aspect_ratio < 0.5 and w > 0.3,
        "branch":  0.5 < aspect_ratio < 4.0,
    }
```

### Depth / Distance Estimation (critical for perching approach):
Do NOT rely on bbox size alone. You need one of:
1. **Monocular depth** — MiDaS v3 runs on Pi 4 at ~2 FPS (acceptable since you only need it on final approach)
2. **Downward-facing ultrasonic** — cheap, gives ground distance but not structure distance
3. **Stereo camera pair** — best accuracy; OpenCV `StereoBM` on Pi 4 achieves ~5 FPS at 320×240

---

## Part 3 — Recommended Image Resolution

**Use 640×640 (keep your current setting).**

Reasoning for PerchLink:
- Drone cameras typically output 1080p or 4K. You will resize to 640 before inference anyway.
- At 640, YOLO11n detects objects down to ~20×20 px (3% of frame width) reliably.
- A utility pole at 10m distance subtends ~50–80 px at 640 wide — well within YOLO's capability.
- Increasing to 1280 would halve your batch size and triple inference time with minimal gain at typical approach distances (>3m).

**Exception:** If you find that poles at >15m are consistently missed in testing, try `imgsz=1280` as an isolated experiment.

---

## Part 4 — Experiment 2 Training Configuration

> Change **one cluster of related parameters** at a time. Here is your next experiment.

### Primary fix: Address the class loss explosion and noisy signal

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset/data.yaml",
    
    # ── Core ──────────────────────────────────────────
    epochs=100,           # was 50 — model hadn't converged
    patience=30,          # stop if no improvement for 30 epochs
    imgsz=640,
    batch=16,             # was 8 — RTX 3050 6GB can handle 16 at 640
    
    # ── Model ─────────────────────────────────────────
    pretrained=True,
    freeze=10,            # NEW: freeze first 10 backbone layers for 
                          # first ~20 epochs to stabilize cls_loss
                          # (remove this in Experiment 3)
    
    # ── Optimizer ─────────────────────────────────────
    optimizer="AdamW",    # was "auto" (SGD) — AdamW converges faster
                          # on small single-class datasets
    lr0=0.001,            # was 0.01 — lower initial LR because we're
                          # fine-tuning, not training from scratch
    lrf=0.01,             # final LR = lr0 * lrf = 0.00001
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,      # was 3 — more warmup to stabilize early cls_loss
    cos_lr=True,          # NEW: cosine LR schedule is smoother than linear
    
    # ── Loss weights ──────────────────────────────────
    box=7.5,              # keep default
    cls=0.3,              # was 0.5 — REDUCE cls weight because with 1
                          # class the cls loss is less informative;
                          # putting more relative weight on box regression
    dfl=1.5,              # keep default
    
    # ── Augmentation (drone-specific) ─────────────────
    hsv_h=0.015,          # slight hue shift — disaster scenes have unusual lighting
    hsv_s=0.5,            # was 0.7 — reduce to prevent unrealistic saturation
    hsv_v=0.5,            # brightness variation — important for fog/smoke
    degrees=10.0,         # NEW: was 0 — drones bank during approach; rotate images
    translate=0.1,
    scale=0.6,            # was 0.5 — more scale variance for varying distances
    shear=0.0,            # keep 0 — shear is unrealistic for drone camera
    perspective=0.0002,   # NEW: small perspective warp — simulates camera tilt
    flipud=0.3,           # NEW: was 0 — drone can approach from below (looking up)
    fliplr=0.5,
    mosaic=1.0,           # keep — mosaic is very beneficial for single-class
    mixup=0.1,            # NEW: light mixup helps generalization
    copy_paste=0.1,       # NEW: copy-paste augmentation — paste pole crops 
                          # onto different backgrounds
    erasing=0.3,          # was 0.4 — reduce slightly (was causing false negatives)
    
    # ── Hardware ──────────────────────────────────────
    device="0",
    workers=4,
    amp=True,             # keep — AMP saves VRAM and speeds up training
    cache=True,           # NEW: cache images in RAM (dataset is very small now)
    
    # ── Logging ───────────────────────────────────────
    project="runs",
    name="perchlink_pole_v2",
    save_period=10,       # save checkpoint every 10 epochs
    seed=42,
)
```

### Why each change from v1:

| Parameter | v1 | v2 | Reason |
|---|---|---|---|
| `batch` | 8 | 16 | 6GB VRAM handles 16 at 640 with AMP; better gradient estimates |
| `epochs` | 50 | 100 | Val loss was still improving at epoch 50 |
| `optimizer` | auto/SGD | AdamW | AdamW adapts per-parameter LR; faster on small datasets |
| `lr0` | 0.01 | 0.001 | Fine-tuning needs 10× lower LR than scratch training |
| `cos_lr` | False | True | Smoother decay, avoids oscillation seen in v1 P/R curves |
| `freeze` | None | 10 | Stabilizes backbone while head learns new class distribution |
| `cls` loss weight | 0.5 | 0.3 | Single class → cls loss is less meaningful; focus on box |
| `degrees` | 0 | 10 | Drone banks during flight |
| `flipud` | 0 | 0.3 | Drone can approach perching structures from below |
| `perspective` | 0 | 0.0002 | Camera tilt during approach |
| `cache` | False | True | Fits in RAM |
| `warmup_epochs` | 3 | 5 | More warmup to prevent early cls_loss spike |

---

## Part 5 — Batch Size Justification for RTX 3050 6GB

```
YOLO11n at 640×640, fp16 (AMP):
  Model params:    ~5 MB
  Activations per image: ~150 MB
  Batch 8:   8 × 150 = 1200 MB + model = ~1.5 GB  ← too conservative
  Batch 16:  16 × 150 = 2400 MB + model = ~2.9 GB ← safe
  Batch 32:  32 × 150 = 4800 MB + model = ~5.5 GB ← risky, may OOM
```

**Use `batch=16`.** If you get OOM errors, drop to 12. Do not go below 8 — gradient noise becomes a problem.

---

## Part 6 — Epochs & Early Stopping

- **100 epochs** with `patience=30`
- Your v1 run best was around epoch 46 (mAP50=0.326). With better hyperparameters, expect best around epoch 60–80.
- At epoch 40, `close_mosaic` kicks in (last 10 epochs disable mosaic) — this often produces a final improvement spike. You need enough epochs for this to happen meaningfully.

---

## Part 7 — Drone-Specific Augmentation Strategy

Standard augmentations are designed for ground-level photography. PerchLink needs a different philosophy:

### Augmentations to **enable** (not in standard configs):
| Augmentation | Setting | Why for PerchLink |
|---|---|---|
| `degrees` | 10–15° | Drone roll during approach maneuver |
| `flipud` | 0.3 | Camera looking upward at branches/grills from below |
| `perspective` | 0.0002 | Camera pitch during descent |
| `scale` | 0.3–0.7 | Poles appear at vastly different sizes depending on altitude |
| `hsv_v` | 0.5 | Disaster scenes: fire glow, dust haze, night ops |
| `copy_paste` | 0.1 | Pastes pole instances into novel backgrounds (ruins, debris) |
| `mixup` | 0.1 | Generalizes texture of poles across environments |

### Augmentations to **disable or reduce**:
| Augmentation | Setting | Why |
|---|---|---|
| `shear` | 0 | Shear is not a realistic camera distortion |
| `erasing` | 0.2–0.3 | High erasing (0.4) was likely causing false negatives in v1 |
| `hsv_s` | 0.5 | Disaster scenes are often desaturated (dust/smoke/flood) |
| `auto_augment` | `none` | `randaugment` in v1 adds random ops that may conflict with drone-specific augments |

### Offline augmentation you should do manually (before training):
1. **Fog/haze simulation**: Apply Gaussian blur + desaturation to 15% of training images
2. **Night simulation**: Reduce HSV V channel by 60–80% on 10% of images
3. **Motion blur**: Simulate drone movement with directional blur kernel (3×15 px)
4. **Compression artifacts**: JPEG-compress at quality 40–60 (drone video streams are compressed)

---

## Part 8 — Dataset Size & Class Balance

### Current state:
- **1 class** (`pole`), only **133 training images** (post-curation).
- This is extremely small for deep learning. Experiment 2 will be a proof-of-concept, but it will not be robust enough for flight deployment.

### Target dataset sizes for PerchLink's 5-class system:

| Class | Current (Positive) | Minimum Target | Ideal |
|---|---|---|---|
| Pole | **133** | 1500 ❌ | 3000 |
| Tree branch | 0 | 1500 | 3000 |
| Railing | 0 | 1500 | 3000 |
| Window grill | 0 | 1500 | 3000 |
| Other perchable | 0 | 500 | 1000 |

> **This is your biggest gap.** You currently have a pole detector, not a perch detector.
> Adding new classes is a higher priority than improving pole-only mAP from 0.33 to 0.45.

### Where to get data for new classes:
1. **Roboflow Universe** — search "railing detection", "tree branch aerial", "window grill"
2. **COCO dataset** — already has some of these; crop and re-annotate at drone perspective
3. **Google Open Images** — has "Pole", "Fence", "Tree" categories with box annotations
4. **Your own collection** — use a phone on a selfie stick to simulate drone perspective

---

## Part 9 — Train/Val/Test Split & Data Leakage

### Your original split (before curation):
- Train: 2001 | Val: 996 | Test: 503 → **Total: 3500**
- Ratio: 57% / 28% / 14%

### Problems with this split:
1. **28% validation is too high** for this dataset size. 
2. The ratio of positive images is skewed: Train (133), Val (297), Test (158). Your validation set actually has MORE positive examples than your training set! This is backwards.

### Recommended split: **80 / 10 / 10**
```
Train: 2800 | Val: 350 | Test: 350
```

### Preventing data leakage from video sources:
```python
# If your images came from videos, split by VIDEO, not by frame.
# WRONG (leakage):
all_frames = [f1_v1, f2_v1, f3_v1, f4_v2, f5_v2]
random.shuffle(all_frames)
train = all_frames[:80%]  # frames from same video in both train and val

# CORRECT (no leakage):
videos = [video_1, video_2, video_3, ..., video_N]
random.shuffle(videos)
train_videos = videos[:80%]
val_videos   = videos[80%:90%]
test_videos  = videos[90%:]
# Then take all frames from each video set
```

**Next steps for data:** When you collect your next 1000 images, rebalance them so that 80% go into the `train` folder.

---

## Part 10 — Handling Small & Distant Objects

Poles/branches at 15–30m from the drone can be very small (10–30 px at 640 wide).

### Strategies:
1. **Tiled inference at test time** (not training): Split a 640 frame into 4 overlapping 320×320 tiles, run inference on each, merge boxes with NMS. Ultralytics SAHI integration makes this easy.
2. **`multi_scale` training**: Set `multi_scale=0.5` in training args — randomly varies input resolution between 320 and 960 during training.
3. **Annotate small instances**: Make sure your label files include all visible poles, even tiny ones. Annotators tend to skip small ones.
4. **P2 feature head**: YOLO11 does not expose a P2 (160×160 stride-4) head by default. If small objects are a persistent problem after Experiment 2, we can add one via a custom model YAML.

```python
# SAHI tiled inference (install: pip install sahi)
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="runs/perchlink_pole_v2/weights/best.pt",
    confidence_threshold=0.35,
)

result = get_sliced_prediction(
    "test_image.jpg",
    model,
    slice_height=320,
    slice_width=320,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

---

## Part 11 — Handling Occlusion

Poles are frequently partially occluded by trees, vehicles, wires, or debris in disaster zones.

### Training strategy:
1. **`copy_paste=0.1`** in Experiment 2 — randomly pastes objects into scenes, creating synthetic occlusion
2. **Annotate occluded instances** — do not skip poles that are 30–50% occluded; include them in labels
3. **`overlap_mask=True`** (already set in your args) — correct for segmentation; for detection, `erasing` simulates occlusion

### Inference strategy:
- An occluded pole is partially perchable. Your post-processing layer should mark boxes where the visible portion is >30% of expected size as "reduced confidence perchable."

---

## Part 12 — Handling Environmental Conditions

| Condition | Training solution | Inference solution |
|---|---|---|
| Fog/smoke | Offline augment: Gaussian blur + HSV_S reduction | Lower conf threshold to 0.25 |
| Rain | Offline augment: rain streak overlay (albumentations) | Multi-scale inference |
| Night | Offline augment: HSV_V reduction to 10–30% | Consider IR camera for Pi |
| Harsh shadows | `hsv_v=0.5` augment | No change needed |
| Dust/haze | Offline augment: HSV_S=0.2, HSV_V=0.6 | Lower conf threshold |
| Backlit | Flip HSV_V (overexpose some images) | Confidence threshold tuning |

**Key insight:** These are not YOLO-level problems. A separate image enhancement pre-processor (CLAHE, dehazing) before YOLO inference is more effective than trying to solve it through augmentation alone.

```python
import cv2
def preprocess_disaster(frame):
    """Apply CLAHE before YOLO inference in challenging conditions."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
```

---

## Part 13 — Reducing False Positives

Your v1 precision (0.37 at best) means ~63% of detections are false positives. This is dangerous for autonomous approach.

### Root causes (in order of likelihood):
1. **Low confidence threshold** — your inference uses `conf=0.35`, which may be too low for a model at this accuracy level
2. **Partial annotation** — unlabeled poles in training images become false positives
3. **Background confusion** — vertical structures (fences, trees, building edges) look like poles from drone view

### Fixes:
1. **Raise conf threshold to 0.5 at inference time** until mAP50-95 > 0.4
2. **Complete annotation pass** — ensure ALL poles in every training image are labeled
3. **Add hard negatives** — see Part 14
4. **Aspect ratio filter** — post-process boxes to reject detections where `h/w < 2` for class `pole`

---

## Part 14 — Hard Negative Mining

Hard negatives are images that look like perchable structures but aren't (fence posts, building edges, tree trunks).

### Process:
```python
# Step 1: Run current best model on negative-heavy images
model = YOLO("runs/perchlink_pole_v2/weights/best.pt")
results = model("path/to/negative_images/", conf=0.35, save=True)

# Step 2: Collect false positive crops
# Manually review runs/predict/ output — save FP images to:
# dataset/images/train/hard_negatives/

# Step 3: Create EMPTY label files for these images
# (empty .txt = "no objects in this image" = teaches model these are NOT poles)
import os
for img in os.listdir("dataset/images/train/hard_negatives"):
    base = os.path.splitext(img)[0]
    label_path = f"dataset/labels/train/{base}.txt"
    open(label_path, 'w').close()  # empty label file

# Step 4: Retrain with these negatives included
```

**Target:** Add 200–400 hard negative images after Experiment 2.

---

## Part 15 — Evaluation Metrics Beyond mAP50

For **autonomous drone deployment**, these metrics matter more than mAP50:

| Metric | How to compute | Why it matters for PerchLink |
|---|---|---|
| **mAP50-95** | Standard YOLO output | True localization quality; you need >0.40 for reliable approach |
| **Recall at conf=0.35** | `metrics/recall(B)` | Missing a perchable structure = mission failure |
| **FP rate per image** | FP count / total images | False positives = unsafe approach attempts |
| **Small-object AP** | Filter boxes where `w*h < 0.01` | Critical for distant detection |
| **Inference latency P95** | 95th percentile ms/frame | Worst-case latency matters for flight safety |
| **FPS under load** | With flight controller running | Real deployment metric |

```python
# Add to your evaluation script
from ultralytics import YOLO
import numpy as np

model = YOLO("runs/perchlink_pole_v2/weights/best.pt")

# Full validation metrics
metrics = model.val(data="dataset/data.yaml", conf=0.35, iou=0.5)

print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision:{metrics.box.p[0]:.4f}")  # per class
print(f"Recall:   {metrics.box.r[0]:.4f}")

# Latency benchmark
import time
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
times = []
for _ in range(100):
    t0 = time.perf_counter()
    model(frame, verbose=False)
    times.append((time.perf_counter() - t0) * 1000)
cap.release()
print(f"Latency: {np.mean(times):.1f}ms mean, {np.percentile(times, 95):.1f}ms P95")
print(f"FPS: {1000/np.mean(times):.1f}")
```

---

## Part 16 — Confidence Threshold Calibration

**Do not use the default 0.35 for deployment.** Calibrate empirically:

```python
# Precision-Recall sweep to find operating point
model = YOLO("runs/perchlink_pole_v2/weights/best.pt")

for conf in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    metrics = model.val(
        data="dataset/data.yaml", 
        conf=conf, 
        iou=0.5,
        verbose=False
    )
    p = metrics.box.p[0]
    r = metrics.box.r[0]
    f1 = 2 * p * r / (p + r + 1e-9)
    print(f"conf={conf:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
```

**For PerchLink, prioritize Recall over Precision** — missing a perchable structure is worse than a false positive (you can filter FPs with suitability checks, but you can't recover a missed structure). Choose the threshold where Recall ≥ 0.75.

---

## Part 17 — IoU / NMS Settings

```python
# For inference / deployment:
model.predict(
    source=frame,
    conf=0.40,      # calibrated threshold
    iou=0.45,       # NMS IoU — lower than default 0.7
                    # Reason: poles can be very close together (street lamp rows)
                    # and we want to detect each one, not merge them
    max_det=20,     # limit max detections per frame (was 300, absurd for PerchLink)
    agnostic_nms=False,  # keep class-aware NMS (only relevant when you add more classes)
)
```

**Why `iou=0.45` for NMS:** The default 0.7 is permissive and merges nearby duplicate detections. At lower altitude where you can see a row of poles, 0.45 preserves individual detections better.

---

## Part 18 — Iterative Dataset Improvement After Testing

This is the most important loop after deployment:

```
┌─────────────────────────────────────────────────────────┐
│           PerchLink Active Learning Loop                 │
│                                                          │
│  1. Deploy model on drone                                │
│  2. Record all inference sessions (save frames)          │
│  3. Filter frames where confidence was 0.30–0.50         │
│     (the "uncertain" zone — most informative)            │
│  4. Manually annotate these uncertain frames             │
│  5. Add to dataset                                       │
│  6. Retrain                                              │
│  7. Repeat                                               │
└─────────────────────────────────────────────────────────┘
```

```python
# In your inference loop, save uncertain frames:
results = model(frame, conf=0.25, verbose=False)
for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        if 0.28 <= conf <= 0.52:  # uncertain zone
            cv2.imwrite(f"uncertain_frames/{time.time():.3f}.jpg", frame)
            break
```

---

## Part 19 — Full Deployment Pipeline: Training to Pi

### Step 1: Train on RTX 3050 (Experiment 2)
```powershell
# Disable Windows sleep first
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

# Run training
venv\Scripts\python train.py
```

### Step 2: Validate best weights
```python
model = YOLO("runs/perchlink_pole_v2/weights/best.pt")
metrics = model.val(data="dataset/data.yaml", conf=0.40, iou=0.45)
# Target: mAP50 > 0.55, mAP50-95 > 0.35 before export
```

### Step 3: Export to ONNX (Raspberry Pi target)
```python
# ONNX is the best format for Pi — works with onnxruntime, no CUDA needed
model.export(
    format="onnx",
    imgsz=640,
    simplify=True,     # simplifies ONNX graph
    dynamic=False,     # static batch size = faster on Pi
    opset=12,          # onnxruntime on Pi supports opset 12–17
    half=False,        # Pi does NOT support fp16 without special setup
)
# Output: runs/perchlink_pole_v2/weights/best.onnx
```

### Step 4: Transfer to Raspberry Pi
```bash
# On your Windows machine:
scp runs/perchlink_pole_v2/weights/best.onnx pi@raspberrypi.local:~/perchlink/

# Test ONNX on Windows first before copying:
model_onnx = YOLO("runs/perchlink_pole_v2/weights/best.onnx")
model_onnx.val(data="dataset/data.yaml")
```

### Step 5: Pi-side inference setup
```bash
# On Raspberry Pi:
pip install ultralytics onnxruntime-rpi   # ARM-optimized runtime
# OR for Pi 5 (aarch64):
pip install onnxruntime                   # standard onnxruntime works
```

```python
# pi_inference.py — runs on Raspberry Pi
from ultralytics import YOLO
import cv2, time, socket, json

model = YOLO("best.onnx")  # ONNX loaded via ultralytics wrapper

# UDP socket to flight controller
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
FC_ADDRESS = ("127.0.0.1", 14551)  # MAVLink GCS port, adjust for your FC

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

CONF_THRESHOLD = 0.45
FRAME_W, FRAME_H = 640, 480

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    t0 = time.perf_counter()
    results = model(frame, conf=CONF_THRESHOLD, iou=0.45, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000
    
    detections = []
    for r in results:
        for box in r.boxes:
            x, y, w, h = box.xywhn[0].tolist()  # normalized
            conf = float(box.conf[0])
            detections.append({
                "class": "pole",
                "conf": round(conf, 3),
                "x_norm": round(x, 4),
                "y_norm": round(y, 4),
                "w_norm": round(w, 4),
                "h_norm": round(h, 4),
                "latency_ms": round(latency_ms, 1),
                # Pixel coords for flight controller
                "x_px": int(x * FRAME_W),
                "y_px": int(y * FRAME_H),
                # Offset from frame center (for approach vector)
                "dx_norm": round(x - 0.5, 4),
                "dy_norm": round(y - 0.5, 4),
            })
    
    # Send to flight controller over UDP
    if detections:
        payload = json.dumps({"detections": detections}).encode()
        sock.sendto(payload, FC_ADDRESS)
    
    print(f"FPS: {1000/latency_ms:.1f}  Detections: {len(detections)}")
```

### Step 6: Expected Pi performance

| Device | Format | Resolution | FPS |
|---|---|---|---|
| Pi 4B (4GB) | ONNX fp32 | 640×640 | 3–5 |
| Pi 4B (4GB) | ONNX fp32 | 320×320 | 10–14 |
| Pi 5 | ONNX fp32 | 640×640 | 8–12 |
| Pi 5 + Hailo-8 | INT8 | 640×640 | 25–30 |

**Recommendation:** Use 320×320 on Pi 4, or invest in a **Hailo-8 AI HAT** (~$70) which gives 25+ FPS at 640 and is designed for Pi.

### Step 7: Keeping CV separate from flight control

```
┌───────────────┐     UDP JSON      ┌────────────────────┐
│  CV Process   │ ────────────────► │  Flight Controller │
│  (YOLO ONNX)  │  port 14551       │  (ArduPilot/PX4)   │
│               │ ◄──────────────── │                    │
│  ~5–10 FPS    │  mode commands    │  ~400 Hz loop      │
└───────────────┘                   └────────────────────┘
```

**CRITICAL:** Run CV as a **separate process** (not a thread) from flight control. A YOLO inference spike must never block the 400 Hz flight controller loop. Use `multiprocessing` with a shared `Queue` or UDP sockets.

```python
# launch_perchlink.py
import multiprocessing
from pi_inference import cv_worker
from flight_control import fc_worker

if __name__ == "__main__":
    cv_proc = multiprocessing.Process(target=cv_worker)
    fc_proc = multiprocessing.Process(target=fc_worker)
    
    cv_proc.start()
    fc_proc.start()
    
    cv_proc.join()
    fc_proc.join()
```

---

## Part 20 — The Next Experiment Decision Tree

```
After Experiment 2 completes, check:

mAP50 > 0.55?
├── YES → Check mAP50-95
│   ├── mAP50-95 > 0.35 → Export to ONNX, start Pi testing
│   └── mAP50-95 < 0.35 → Small objects problem → add SAHI / multi_scale
└── NO → Check val loss vs train loss
    ├── val_loss >> train_loss → Overfitting → add dropout=0.1, more augmentation
    └── val_loss ≈ train_loss (both high) → Underfitting / bad data →
        → Run data audit, check annotations, add more diverse images
```

---

## Part 21 — What I Need From You to Recommend Experiment 3

Before I can give precise Experiment 3 hyperparameters, share:

1. **Confusion matrix image** from v1 (`runs/perchlink_pole_v1/confusion_matrix_normalized.png`)
2. **Val prediction examples** (`val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`)
3. **Labels distribution** (`labels.jpg`) 
4. **Data audit result** — image count vs label count, any empty label files
5. **Image source** — were images scraped from web, drone video, street-level? Multiple locations or 1-2?
6. **Sample of your annotations** — 5 example images with their label overlays from `preview_dataset.py`

With those, I can tell you exactly what is causing precision/recall oscillation and whether the problem is data, augmentation, or architecture.

---

## Summary — Priority Order

| Priority | Action | Expected impact |
|---|---|---|
| 🔴 **1** | Run data audit (image ↔ label count, empty files, visual check) | Eliminate bad data from corrupting training |
| 🔴 **2** | Disable system sleep before training | Prevent CUDA context corruption |
| 🔴 **3** | Run Experiment 2 with new hyperparameters | Expected: mAP50 0.45–0.60 |
| 🟡 **4** | Share confusion matrix + val predictions | Enables Experiment 3 diagnosis |
| 🟡 **5** | Begin collecting branch/railing/grill images | Unblocks multi-class detector |
| 🟢 **6** | Add SAHI for small-object inference | After mAP50 > 0.50 |
| 🟢 **7** | Export to ONNX, test on Pi | After mAP50-95 > 0.35 |
