# PerchLink CV — Disaster-Zone SOS Relay Drone Vision System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Model Architecture](https://img.shields.io/badge/Architecture-Ultralytics%20YOLO11n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Target Platform](https://img.shields.io/badge/Target%20Edge%20Platform-Raspberry%20Pi%205%20(1%20GB%20RAM)-orange.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Dataset](https://img.shields.io/badge/Dataset-HOT--OSM%20Street--Level%20Poles-purple.svg)](https://huggingface.co/datasets/hotosm/streetlevel-poles)
[![Baseline Metric](https://img.shields.io/badge/Exp%20C%20mAP%4050-0.5897-brightgreen.svg)]()

> **PerchLink** is an autonomous drone concept engineered to deliver persistent communications connectivity in disaster zones (such as post-earthquake or flood environments) where cellular towers and power grids have collapsed. Rather than burning battery energy hovering, the drone flies to a disaster area, visually identifies an upright structural utility pole or tower, perches onto it, powers down its propulsion motors, and operates for hours as a stationary, low-power SOS mesh relay node.
>
> **Scope Notice:** This repository houses the **computer-vision (CV) subsystem** for PerchLink — responsible for utility pole detection, visual position guidance, dataset curation, and edge model optimization. It does *not* contain flight-control avionics, motor firmware, physical gripper actuation, or mesh RF payload hardware.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current Computer-Vision Objective](#current-computer-vision-objective)
3. [Verified Features](#verified-features)
4. [Technical Stack](#technical-stack)
5. [Architecture & Workflow](#architecture--workflow)
6. [Repository Structure](#repository-structure)
7. [Installation & Setup](#installation--setup)
8. [CLI Usage & Verified Commands](#cli-usage--verified-commands)
9. [Dataset & Preparation](#dataset--preparation)
10. [Training & Experimental Evaluation](#training--experimental-evaluation)
11. [Raspberry Pi 5 Deployment Strategy](#raspberry-pi-5-deployment-strategy)
12. [Safety Boundaries, Limitations & Roadmap](#safety-boundaries-limitations--roadmap)
13. [Credits & Attribution](#credits--attribution)

---

## Project Overview

In severe disaster response scenarios, commercial drones face a critical limitation: rotor hover flight time is typically capped at 25–45 minutes. Continuous hovering depletes battery life rapidly, rendering aerial communications relays unsustainable for ongoing emergency coordination.

```
       [Disaster Zone / Search & Rescue]
                       │
                       ▼
 ┌──────────────────────────────────────────┐
 │      PerchLink Aerial Deployment         │
 │  - Search phase: scan for utility poles  │
 │  - Vision pipeline detects candidates    │
 └─────────────────────┬────────────────────┘
                       │
                       ▼
 ┌──────────────────────────────────────────┐
 │     Visual Alignment & Approach          │
 │  - Maintain optical center (LEFT/CTR/RT) │
 │  - Rough monocular range monitoring      │
 └─────────────────────┬────────────────────┘
                       │
                       ▼
 ┌──────────────────────────────────────────┐
 │       Perching & Motor Shutdown          │
 │  - Mechanical gripper secures to pole    │
 │  - Rotors power OFF (energy conserved)   │
 └─────────────────────┬────────────────────┘
                       │
                       ▼
 ┌──────────────────────────────────────────┐
 │     Persistent Low-Power SOS Relay       │
 │  - Hours of mesh networking & beacons    │
 └──────────────────────────────────────────┘
```

The computer-vision subsystem documented here provides the situational awareness needed during the approach phase: locating utility poles in street and low-altitude imagery, maintaining centering relative to the drone's optical axis, and generating edge-efficient detection models.

---

## Current Computer-Vision Objective

The current milestone focuses on **single-class utility pole and vertical infrastructure detection** using high-efficiency deep convolutional/attention models capable of running on severely constrained edge hardware.

### Capability Matrix

| Status | Capability | Implementation / Notes |
|---|---|---|
| **Implemented & Verified** | Single-class utility pole detection | Fine-tuned Ultralytics YOLO11n model via consolidated `perchlink_cv.py` |
| **Implemented & Verified** | Batch & single-image inference | CLI `predict` subcommand with custom thresholds, labels, and directory recursion |
| **Implemented & Verified** | Live camera inference loop | CLI `live` subcommand with real-time OpenCV stream processing and graceful shutdown |
| **Implemented & Verified** | Optical center position guidance | Real-time classification into `LEFT`, `CENTER`, and `RIGHT` zones with 100px deadband |
| **Implemented & Verified** | Monocular pinhole range estimation | Geometric range calculation ($Z \approx f \cdot D / d$) logged to CSV |
| **Implemented & Verified** | Ground-truth bench error logging | Dual logging of tape-measured distance, absolute error, and percentage error |
| **Implemented & Verified** | Dataset audit & annotation sanity | CLI `audit` subcommand checking IoU duplicates, bounds, aspect ratios, and empty files |
| **Implemented & Verified** | Dataset background curation | CLI `curate` subcommand balancing excessive empty images with opt-in safeguard |
| **Implemented & Verified** | Stratified dataset re-splitting | CLI `split` subcommand partitioning data into train/val/test with fixed seed |
| **Implemented & Verified** | Parquet-to-YOLO converter | CLI `convert` subcommand parsing HOT-OSM Apache Parquet data |
| **Implemented & Verified** | Model export pipeline | CLI `export` subcommand supporting ONNX, NCNN, TFLite, and TorchScript |
| **Experimentally Evaluated** | High-resolution transfer learning | Experiment C (1024px, 100 epochs, backbone frozen): **0.5897 mAP@0.50** on validation split |
| **Planned / Future** | Multi-class pole & crossbar geometry | Distinguishing wooden poles, concrete towers, metal masts, and crossbeams |
| **Planned / Future** | Oriented Bounding Boxes (OBB) | Accurate angle estimation for tilted poles or horizontal perching bars |
| **Planned / Future** | MAVLink flight controller integration | Transmitting vision offsets to BotStack BotWing F405 flight controller via UART |
| **Planned / Future** | Physical Raspberry Pi 5 benchmarking | Measuring FPS, memory usage, and thermal throttling under active 1 GB RAM limit |

> [!CAUTION]
> **Safety Disclaimer:** Visual detection of a pole does **not** indicate or guarantee that the structure is mechanically sound, uncompromised by disaster damage, free of live high-voltage wiring, or safe for drone perching or landing.

---

## Verified Features

All maintained project workflows are consolidated within a **single executable entry point**: `perchlink_cv.py`.

* **Consolidated CLI Architecture:** Ten dedicated subcommands (`live`, `predict`, `train`, `evaluate`, `export`, `convert`, `audit`, `preview`, `curate`, `split`) with consistent argument naming and POSIX/Windows CLI support.
* **Strict Model Safety Check:** The CLI rejects missing checkpoints and explicitly forbids silent fallback to generic COCO weights. Live inference requires a verified fine-tuned pole detector.
* **Optical Position Guidance:** Evaluates detection centroid relative to the video frame's horizontal center ($W/2$). Identifies targets as `LEFT`, `RIGHT`, or aligned `CENTER` (within a configurable $\pm 100\text{ px}$ deadband) for drone visual servoing.
* **Pinhole Monocular Range Estimation:**
  $$\text{Distance } (Z) \approx \frac{f_{\text{px}} \times \text{Target Dimension (m)}}{\text{Observed Dimension (px)}}$$
  Requires calibrated field of view (`--hfov-deg` or `--focal-length-px`) and user-entered target dimension (`--target-size-m`).
* **Bench-Test Validation Logging:** Automatically calculates $|Z_{\text{pred}} - Z_{\text{truth}}|$ and percentage error, appending structured records to `runs/perchlink_range_estimates.csv`.
* **Drone-Specific Data Augmentations:** Configurable affine scale, HSV jitter, mosaic, mixup, and random erasing. Vertical flip (`flipud`) is intentionally locked to `0.0` by default to preserve the upright gravity orientation of utility poles.

---

## Technical Stack

### Development & Training Workstation
* **Operating System:** Microsoft Windows 11 Home (x64)
* **Python Runtime:** Python 3.14.2 (virtual environment: `venv`)
* **Deep Learning Framework:** PyTorch 2.11.0 + CUDA 12.8 (`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`)
* **GPU Acceleration:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA Compute Capability 8.6, 6144 MiB VRAM)
* **Computer Vision Library:** OpenCV 5.0.0 (`opencv-python==5.0.0.93`)
* **Model Framework:** Ultralytics YOLO 8.4.171
* **Data Processing:** pandas 3.0.6, pyarrow 25.0.1, pillow 12.3.0, PyYAML 6.0.3, tqdm 4.70.1

### Target Edge Deployment Platform
* **Target Hardware:** Raspberry Pi 5 (Single Board Computer)
* **CPU:** Broadcom BCM2712 quad-core ARM Cortex-A76 @ 2.4 GHz
* **System Memory:** **1 GB LPDDR4X SDRAM** (extremely tight memory envelope)
* **Target Runtime:** Linux (Raspberry Pi OS Bookworm 64-bit), ONNX Runtime / NCNN engine
* **Camera Interface:** Raspberry Pi Camera Module 3 or USB UVC feed

---

## Architecture & Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PerchLink CV Pipeline                           │
└────────────────────────────────────────────────────────────────────────┘

 1. INGESTION       Hugging Face: hotosm/streetlevel-poles (Parquet)
       │
       ▼
 2. CONVERSION      perchlink_cv.py convert --source dataset/hotosm
       │            Extracts embedded bytes -> dataset/images & labels
       ▼
 3. AUDIT & SANITY  perchlink_cv.py audit --dataset dataset
       │            Catches corrupted coordinates, duplicates, out-of-bounds
       ▼
 4. CURATION        perchlink_cv.py curate --bg-ratio 0.10 --apply
       │            Filters out excessive empty frames (93% background -> 10%)
       ▼
 5. PARTITIONING    perchlink_cv.py split --train-ratio 0.8 --val-ratio 0.1 --apply
       │            Stratified split: 516 Train / 64 Val / 65 Test
       ▼
 6. FINE-TUNING     perchlink_cv.py train --model yolo11n.pt --imgsz 1024 --freeze 10
       │            100 epochs, AdamW, cosine LR, backbone frozen
       ▼
 7. EVALUATION      perchlink_cv.py evaluate --model runs/.../best.pt --split val
       │            Verified Metric: 0.5897 mAP@50 | 0.2097 mAP@50-95
       ▼
 8. RUNTIME INFER   perchlink_cv.py live --model runs/.../best.pt --camera 0
       │            Real-time detection + LEFT/CENTER/RIGHT alignment guidance
       ▼
 9. EDGE EXPORT     perchlink_cv.py export --model runs/.../best.pt --format onnx
                    Prepares lightweight graph for Raspberry Pi 5 (1 GB RAM)
```

---

## Repository Structure

```text
perchlink_cv/
├── .gitignore                          # Protective Git rules (ignores large weights, videos, datasets)
├── requirements.txt                    # Audited and portable dependency manifest
├── perchlink_cv.py                     # SOLE maintained unified Python entry point & library
├── perchlink_ml_pipeline.md            # In-depth ML pipeline strategy & experimental analysis
├── README.md                           # Technical documentation & usage guide
│
├── dataset/
│   ├── data.yaml                       # Primary single-class YOLO dataset configuration
│   ├── data_aerial_combined.yaml       # Multi-source dataset configuration (street + aerial)
│   ├── data_aerial_eval.yaml           # Dedicated aerial evaluation configuration
│   ├── data_aerial_v2.yaml             # Hard-negative augmented aerial configuration
│   └── hotosm/
│       ├── .gitattributes              # Hugging Face LFS configuration
│       ├── README.md                   # Upstream HOT-OSM dataset metadata & citation
│       └── data.yaml                   # Upstream dataset configuration
│
├── runs/
│   └── perchlink_range_estimates.csv   # Bench-testing range validation log (8,600+ rows)
│
└── test_media/
    └── images/                         # Small static test imagery for local verification
        ├── Nine-Scene Pole-Free Landscape Collage.png
        ├── Screenshot 2026-10-23 144033.png
        ├── Screenshot 2026-10-23 144454.png
        └── Screenshot 2026-10-23 144913.png
```

### Local-Only & Excluded Artifacts (Not in Git)
To adhere to GitHub best practices and avoid repository bloat, the following local files and directories are deliberately excluded via `.gitignore`:
* **Virtual Environment:** `venv/`, `.venv/`
* **Large Model Weights (`*.pt`):** `runs/**/best.pt`, `runs/**/last.pt`, `yolo11n.pt`, `yolo11s.pt` (~2.4 GB)
* **Raw & Processed Datasets:** `dataset/images/`, `dataset/labels/`, `dataset/aerial_epri/`, `dataset/hotosm/data/` (~2.7 GB)
* **Bulky Training Runs:** `runs/experiment_*`, `runs/perchlink_*` (plots, batches, logs)
* **High-Resolution Test Videos:** `test_media/videos/*.mp4` (>300 MB)
* **Historical Script Backups:** `backups/` and `archive/`

---

## Installation & Setup

### Prerequisites
* Windows 10/11 or Linux (Ubuntu 22.04+ / Debian 12 / Raspberry Pi OS 64-bit)
* Python 3.10 to 3.14
* Recommended for training: NVIDIA GPU with CUDA 12+ support

### 1. Clone the Repository
```powershell
git clone https://github.com/jamdadeviraj410-oss/perchlink_cv.git
cd perchlink_cv
```

### 2. Create and Activate Virtual Environment
On Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

#### Option A: Workstation with NVIDIA CUDA GPU (Recommended for Training)
```powershell
# 1. Install PyTorch with CUDA 12.8 wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install PerchLink dependencies
pip install -r requirements.txt
```

#### Option B: CPU Only or Raspberry Pi 5 (Edge Deployment)
```bash
# 1. Standard PyTorch (CPU)
pip install torch torchvision

# 2. Install PerchLink dependencies
pip install -r requirements.txt
```

### 4. Verify Installation
Run this quick diagnostic to confirm your environment:
```powershell
python -c "import torch, ultralytics, cv2; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Ultralytics: {ultralytics.__version__} | OpenCV: {cv2.__version__}')"
```

---

## CLI Usage & Verified Commands

All operations run through `perchlink_cv.py`.

### 1. Model Inference on Static Images (`predict`)
```powershell
# Predict on a single image
python perchlink_cv.py predict test_media/images/Screenshot 2026-10-23 144033.png --model runs/experiment_c_resolution/weights/best.pt --conf 0.35

# Predict on an entire folder with YOLO text annotations saved
python perchlink_cv.py predict test_media/images/ --model runs/experiment_c_resolution/weights/best.pt --conf 0.35 --save-txt --save-conf
```

### 2. Live Camera Inference & Guidance (`live`)
Connect a USB webcam or pass a camera index / video stream:
```powershell
# Live detection with default webcam (camera index 0)
python perchlink_cv.py live --model runs/experiment_c_resolution/weights/best.pt --camera 0 --conf 0.35

# Live detection with monocular range estimation (target diameter = 0.20m, camera HFOV = 78 deg)
python perchlink_cv.py live --model runs/experiment_c_resolution/weights/best.pt --camera 0 --target-size-m 0.20 --range-dimension width --hfov-deg 78.0

# Bench validation mode: record ground-truth tape distance for error analysis
python perchlink_cv.py live --model runs/experiment_c_resolution/weights/best.pt --camera 0 --target-size-m 0.20 --hfov-deg 78.0 --validation-distance-m 1.50
```

### 3. Model Training (`train`)
Fine-tune YOLO11n with high resolution and frozen backbone:
```powershell
python perchlink_cv.py train --data dataset/data.yaml --model yolo11n.pt --imgsz 1024 --batch 8 --epochs 100 --freeze 10 --optimizer AdamW --name experiment_c_resolution
```

### 4. Model Evaluation & Threshold Sweeps (`evaluate`)
```powershell
# Evaluate on the validation split
python perchlink_cv.py evaluate --data dataset/data.yaml --model runs/experiment_c_resolution/weights/best.pt --split val --imgsz 1024

# Evaluate on the test split with full threshold sweep
python perchlink_cv.py evaluate --data dataset/data.yaml --model runs/experiment_c_resolution/weights/best.pt --split test --imgsz 1024 --sweep
```

### 5. Dataset Quality Audit (`audit`)
Audits bounding-box coordinates for duplicates, degenerative boxes ($w \le 0$ or $h \le 0$), out-of-bounds coords ($> 1.0$), and extreme aspect ratios:
```powershell
python perchlink_cv.py audit --dataset dataset
```

### 6. Dataset Visual Preview (`preview`)
Preview ground-truth bounding boxes on images without modifying files:
```powershell
python perchlink_cv.py preview --dataset dataset --split train --count 20 --save-dir preview_output/
```

### 7. Background Curation (`curate`)
Reduce excessive empty background images (dry-run by default):
```powershell
# Dry run: check how many images would be modified
python perchlink_cv.py curate --dataset dataset --split train --bg-ratio 0.10

# Opt-in apply: apply deletions to maintain a 10% background ratio
python perchlink_cv.py curate --dataset dataset --split train --bg-ratio 0.10 --apply
```

### 8. Dataset Re-Splitting (`split`)
Re-split curated images into 80% train, 10% validation, 10% test:
```powershell
# Dry run
python perchlink_cv.py split --dataset dataset --train-ratio 0.80 --val-ratio 0.10 --test-ratio 0.10

# Opt-in apply
python perchlink_cv.py split --dataset dataset --train-ratio 0.80 --val-ratio 0.10 --test-ratio 0.10 --apply
```

### 9. Raw Dataset Conversion (`convert`)
Convert raw HOT-OSM parquet files downloaded from Hugging Face into YOLO structure:
```powershell
python perchlink_cv.py convert --source dataset/hotosm --dest dataset
```

### 10. Model Export for Edge Inference (`export`)
Export trained weights to ONNX or NCNN for deployment:
```powershell
# Export to FP16 ONNX with simplified graph
python perchlink_cv.py export --model runs/experiment_c_resolution/weights/best.pt --format onnx --imgsz 640 --half --simplify

# Export to NCNN for ARM / Raspberry Pi
python perchlink_cv.py export --model runs/experiment_c_resolution/weights/best.pt --format ncnn --imgsz 640
```

---

## Dataset & Preparation

### Upstream Source
* **Repository:** [Hugging Face — `hotosm/streetlevel-poles`](https://huggingface.co/datasets/hotosm/streetlevel-poles)
* **Curator:** UC Berkeley / Humanitarian OpenStreetMap Team (HOT)
* **License:** Creative Commons Attribution-ShareAlike 4.0 International ([CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/))

### Dataset Structure & Curation History

The original dataset release contains 3,500 street-level images with annotations for `0: pole` and `1: tower`:

| Split | Raw Images | Positive Images | Negative / Background Images | Total Boxes |
|---|---|---|---|---|
| **Train** | 2,001 | 227 (11.3%) | 1,774 (88.7%) | 338 |
| **Validation** | 996 | 501 (50.3%) | 495 (49.7%) | 816 |
| **Test** | 503 | 256 (50.9%) | 247 (49.1%) | 409 |
| **Total** | 3,500 | 984 (28.1%) | 2,516 (71.9%) | 1,563 |

#### The Background Imbalance Problem & Resolution
In initial training experiments (`v1 baseline`), the model suffered severe convergence failure, with training classification loss starting at an abnormal $\text{cls\_loss} = 29.0$. An audit revealed that **93% of the initial training images contained zero pole bounding boxes** (1,868 empty frames vs 133 positive frames). This severe imbalance caused the network to collapse toward predicting empty frames.

To fix this:
1. `perchlink_cv.py curate` was executed to prune excessive background images to a controlled **10% background ratio**.
2. `perchlink_cv.py split` re-partitioned the curated pool with an 80/10/10 stratified split resulting in **516 train, 64 validation, and 65 test images**.

### Local Dataset Setup Instructions
Because the image datasets exceed 2 GB, they are not committed to Git. To prepare them locally:
1. Download the parquet archives from Hugging Face:
   ```bash
   # Using git-lfs or huggingface-cli
   huggingface-cli download hotosm/streetlevel-poles --repo-type dataset --local-dir dataset/hotosm
   ```
2. Convert and curate using `perchlink_cv.py`:
   ```powershell
   python perchlink_cv.py convert --source dataset/hotosm --dest dataset
   python perchlink_cv.py curate --dataset dataset --bg-ratio 0.10 --apply
   python perchlink_cv.py split --dataset dataset --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --apply
   ```

---

## Training & Experimental Evaluation

### Checkpoint Integrity Verification
The primary historical training checkpoint from **Experiment C** is verified by SHA-256 cryptographic hash:
```
Target Artifact : runs/experiment_c_resolution/weights/best.pt
File Size       : 5,527,706 bytes (~5.27 MB)
SHA-256 Hash    : 0CD57BE496CEB2A3E2622E70982C8793B5AB540F0C2DF65269744F90B36CDA82
Status          : VERIFIED & CONFIRMED
```

### Experiment C Training Configuration
* **Base Architecture:** YOLO11n (`yolo11n.pt`, 2.58M parameters, 6.4 GFLOPs)
* **Input Resolution:** $1024 \times 1024$ pixels (`imgsz=1024`)
* **Batch Size:** 8 (optimized for 6 GB VRAM on RTX 3050)
* **Optimizer:** AdamW with cosine learning rate schedule (`cos_lr=True`)
* **Backbone Freeze:** First 10 layers frozen (`freeze=10`) to preserve low-level spatial features and prevent overfitting on small positive samples
* **Epochs & Early Stopping:** 100 epochs with patience 30

### Historical Evaluation Results

> [!NOTE]
> These metrics represent historical evaluation results on the verified validation split under controlled workstation conditions. They are **not** guaranteed performance metrics in unstructured disaster environments and do not represent a production-certified model.

#### Validation Split Performance (64 images, 87 ground-truth instances)
* **mAP@0.50:** **0.5897** (0.58965)
* **mAP@0.50:0.95:** **0.2097** (0.20973)
* **Precision:** **0.6190**
* **Recall:** **0.5520**
* **Inference Speed:** ~15.7 ms / frame on RTX 3050 GPU (batch size 1)

#### Test Split Performance (65 images, 106 ground-truth instances)
* **mAP@0.50:** **0.4493**
* **mAP@0.50:0.95:** **0.1657**
* **Precision:** **0.5590**
* **Recall:** **0.5090**

---

## Raspberry Pi 5 Deployment Strategy

The ultimate operational target for PerchLink is a **Raspberry Pi 5 with only 1 GB RAM**. This presents severe edge deployment constraints:

```
┌────────────────────────────────────────────────────────┐
│             Raspberry Pi 5 (1 GB RAM Budget)            │
├──────────────────────────┬─────────────────────────────┤
│ OS + Kernel + Daemons    │ ~350 - 450 MB               │
│ Vision Runtime Headroom  │ ~300 - 400 MB MAX           │
│ System Reserve / Buffers │ ~150 - 200 MB               │
└──────────────────────────┴─────────────────────────────┘
```

### Edge Optimization Protocol
1. **Model Pruning & Architecture Selection:** YOLO11n (2.6M params) was selected over YOLO11s (9.4M params) and YOLO11m (20.1M params) because larger variants risk Linux Out-Of-Memory (`oom-killer`) termination when allocating PyTorch tensor buffers on 1 GB hardware.
2. **PyTorch Runtime Decoupling:** Running PyTorch (`torch`) directly on the Pi consumes substantial base memory (~400 MB resident memory just importing `torch`). The model must be exported via `perchlink_cv.py export --format onnx` or `--format ncnn` and executed using `onnxruntime` or `ncnn` C++ / lightweight Python bindings.
3. **Resolution Trade-Off:** While workstation training benefited from $1024\times 1024$ input resolution (`0.5897 mAP@50`), edge inference on the Pi 5 will downscale frames to $480\times 480$ or $640\times 640$ to sustain $\ge 8\text{ FPS}$ within memory limits.
4. **Hardware Validation Requirement:** Real-time throughput, thermal throttling, and camera ingestion latency must be empirically measured on the physical device prior to flight-readiness certification.

---

## Safety Boundaries, Limitations & Roadmap

### Clear Subsystem Boundaries
The PerchLink project enforces strict decoupling between system layers:
* **In Scope (This Repo):** Pole detection, visual centering (`LEFT` / `CENTER` / `RIGHT`), bounding box logging, and edge model optimization.
* **Out of Scope (Separate Subsystems):** Flight stabilization, trajectory planning, motor cutoff, gripper closing mechanisms, battery management, and RF mesh communications.

### Key Technical Limitations
1. **Vertical Geometry vs Horizontal Bars:** The detector is trained specifically on vertical infrastructure. For vertical poles, bounding-box width corresponds to diameter. For horizontal pipes, crossbars, or wires, bounding-box width corresponds to the longitudinal span across the camera view, meaning **monocular range cannot be derived from horizontal bar width**.
2. **Pinhole Distance Approximations:** Monocular range based on pinhole geometry is an auxiliary estimation tool for bench experiments. It is **not safety-critical**, cannot replace LiDAR or stereo depth sensors, and must not be used as the sole input for physical docking.
3. **Viewpoint Sensitivity:** Training data predominantly consists of street-level vehicle and pedestrian camera angles. Aerial high-angle viewpoints (looking down from 15–30 meters) require additional aerial dataset fine-tuning.

### Development Roadmap
- [x] Consolidate codebase into unified `perchlink_cv.py` CLI.
- [x] Fix 93% background imbalance via dataset curation and 80/10/10 re-split.
- [x] Train and validate Experiment C baseline (0.5897 mAP@50).
- [x] Implement real-time position guidance (`LEFT`, `CENTER`, `RIGHT`).
- [x] Implement pinhole monocular range logging and bench validation error tracking.
- [x] Integrate multi-format export pipeline (ONNX, NCNN).
- [ ] Incorporate high-angle aerial datasets (e.g. EPRI transmission line imagery) to improve top-down view recall.
- [ ] Evaluate Oriented Bounding Boxes (YOLO11-OBB) for angled poles and crossbar perching structures.
- [ ] Benchmark latency and memory usage on physical Raspberry Pi 5 1 GB hardware using NCNN / ONNX Runtime.
- [ ] Interface vision offset telemetry with BotStack BotWing F405 flight controller via MAVLink over UART.
- [ ] Define multi-sensor safety verification protocols before executing live perching trials.

---

## Credits & Attribution

* **Dataset:** [HOT-OSM Street-level Poles & Towers](https://huggingface.co/datasets/hotosm/streetlevel-poles) curated by UC Berkeley and the Humanitarian OpenStreetMap Team under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
* **Detection Architecture:** [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) by Ultralytics Inc.
* **Libraries:** PyTorch, OpenCV, Pandas, PyArrow, NumPy, Pillow.
* **Project Author:** Viraj Jamdade (`jamdadeviraj410-oss`)

---

### License Note
This repository contains original computer-vision code and documentation for the PerchLink research project. Code in this repository is currently released under All Rights Reserved by the author unless an explicit open-source license is designated in a future release. Third-party dependencies and datasets remain governed by their respective upstream licenses (e.g., CC-BY-SA 4.0 for the HOT-OSM dataset; AGPL-3.0 / Enterprise for Ultralytics).
