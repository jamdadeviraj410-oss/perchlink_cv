#!/usr/bin/env python3
"""
PerchLink CV — Unified Computer Vision Pipeline
================================================
Consolidated single-source-of-truth CLI and Python library for the PerchLink
autonomous drone vision system.

Hardware constraints & targets:
  - Training Workstation : RTX 3050 (6 GB VRAM), 16 GB System RAM, Windows 11
  - Edge Deployment      : Raspberry Pi 5 (1 GB RAM total, ARM Cortex-A76)
  - Flight Controller    : BotStack BotWing F405 (external interface)

Supported Subcommands:
  - live       : Live webcam/camera inference loop with position guidance & monocular range
  - predict    : Inference on static images or directories
  - train      : Model training with pre-flight checks and drone augmentations
  - evaluate   : Validation / test evaluation with threshold sweeps (P, R, F1, mAP)
  - export     : Export PyTorch checkpoint to ONNX / NCNN for edge deployment
  - convert    : Convert raw HOTOSM dataset (Parquet) to YOLO format
  - audit      : Comprehensive dataset and annotation quality audit
  - preview    : Visual inspection of dataset images and ground truth bounding boxes
  - curate     : Balance positive and background images (opt-in deletion)
  - split      : Re-split dataset into train/val/test partitions (opt-in move)

Safety Policy & Subsystem Boundaries:
  - Destructive operations (curate, split, convert --clean) require explicit opt-in flags.
  - Zero side-effects on import or running unrelated subcommands.
  - Live inference strictly defaults to fine-tuned checkpoint:
      runs/experiment_c_resolution/weights/best.pt
  - Never silently falls back to generic COCO weights.
  - Subsystems are strictly decoupled: Gripper actuation and flight control commands
    are separate subsystems and are NOT implemented in this vision pipeline.

Pinhole Monocular Range & Perch-Scene Geometry Limitation:
  - Supports approximate camera-to-target distance via pinhole relation:
      Z ≈ f_px * known_target_dimension_m / observed_dimension_px
  - Requires user-entered physical target dimension (--target-dim-m) and camera calibration
    (either --hfov-deg or --focal-length-px). Dimensions are never guessed from model outputs.
  - DISCLAIMER: Monocular range is purely approximate and NOT safety-rated or LiDAR-equivalent.
  - Perch-Scene Limitation: The frozen detector is a single-class vertical pole detector.
    Bounding-box width corresponds to diameter ONLY for vertical poles. For horizontal bars
    or pipes, box width spans the longitudinal bar extent across the frame, which CANNOT be
    used to estimate distance from bar diameter.
  - Bench-Test Validation: Optional --validation-distance-m records tape-measured ground truth,
    absolute error (m), and percentage error (%) directly into the same CSV row.
  - Observations are appended to a single CSV: runs/perchlink_range_estimates.csv.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import math
import os
import pathlib
import random
import shutil
import signal
import sys
import time
from collections import defaultdict
from typing import Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("perchlink_cv")

# Base directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_YAML = os.path.join(SCRIPT_DIR, "dataset", "data.yaml")
DEFAULT_BEST_CHECKPOINT = os.path.join(
    SCRIPT_DIR, "runs", "experiment_c_resolution", "weights", "best.pt"
)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ==============================================================================
# Helper Utilities
# ==============================================================================

def validate_model_path(model_path: str, context: str = "model") -> str:
    """Validate that the given model path exists.
    
    CRITICAL: Never silently fall back to generic COCO weights.
    """
    abs_path = os.path.abspath(model_path)
    if not os.path.isfile(abs_path):
        err_msg = (
            f"Required {context} checkpoint not found: '{abs_path}'.\n"
            f"Silent fallback to generic COCO weights is disabled.\n"
            f"Please specify a valid fine-tuned checkpoint via --model."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    return abs_path


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IoU) for two YOLO boxes (xc, yc, w, h)."""
    x1_min = box1[0] - box1[2] / 2.0
    x1_max = box1[0] + box1[2] / 2.0
    y1_min = box1[1] - box1[3] / 2.0
    y1_max = box1[1] + box1[3] / 2.0

    x2_min = box2[0] - box2[2] / 2.0
    x2_max = box2[0] + box2[2] / 2.0
    y2_min = box2[1] - box2[3] / 2.0
    y2_max = box2[1] + box2[3] / 2.0

    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def estimate_monocular_range(
    pixel_size: float,
    known_size_m: Optional[float],
    focal_length_px: Optional[float],
    min_pixel_size: float = 8.0,
) -> Tuple[Optional[float], str]:
    """Estimate camera-to-target range from a known target dimension using pinhole geometry:
        Z ≈ f_px * known_target_dimension_m / observed_dimension_px

    DISCLAIMER: This pinhole monocular estimate is purely approximate. It is NOT
    safety-rated, NOT LiDAR-equivalent, and must NOT be used for flight control,
    docking, or automated gripper actuation. The gripper is a separate subsystem.

    PERCH-SCENE LIMITATION (Horizontal Bars vs. Vertical Poles):
    The frozen Phase 5 detector was trained exclusively as a single-class vertical
    pole detector. For vertical poles, the bounding-box width corresponds to the
    pole's physical diameter. However, for horizontal bars or pipes, the bounding-box
    width represents the longitudinal span across the frame (which varies with field
    of view, viewing angle, and frame cropping), NOT its diameter. Do NOT use box
    width with horizontal bar diameter. Estimating distance to horizontal bars requires
    measuring the transverse thickness or an oriented-box/laser-ranging method.

    Returns:
        (distance_m, status_string)
        If range cannot be reliably computed, distance_m is None and status_string
        provides the exact operational reason.
    """
    if known_size_m is None and focal_length_px is None:
        return None, "missing camera calibration and target dimension"
    if known_size_m is None:
        return None, "missing target dimension: provide --target-dim-m"
    if focal_length_px is None:
        return None, "missing camera calibration: provide --hfov-deg or --focal-length-px"
    if known_size_m <= 0:
        return None, "invalid target dimension: must be > 0 m"
    if focal_length_px <= 0:
        return None, "invalid focal length: must be > 0 px"
    if pixel_size <= 0:
        return None, "invalid pixel dimension: must be > 0 px"
    if pixel_size < min_pixel_size:
        return None, f"target dimension {pixel_size:.1f} px under {int(min_pixel_size)} px; range too unreliable"

    approx_range_m = (focal_length_px * known_size_m) / float(pixel_size)
    return approx_range_m, "approximate (pinhole estimate, non-safety-rated)"


# ==============================================================================
# Subcommand 1: LIVE (Webcam / Camera feed inference)
# ==============================================================================

def ensure_range_log_file(csv_path: str) -> None:
    """Ensure runtime range CSV file exists with the standard 16-column header.
    
    Preserves existing rows, existing columns, and append-only semantics.
    If an older 13-column file is encountered, upgrades the header and pads
    existing data rows without data loss.
    """
    columns = [
        "timestamp",
        "frame",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "chosen_pixel_dimension",
        "entered_real_world_dimension_m",
        "calibration_value",
        "approx_range_m",
        "status",
        "validation_distance_m",
        "abs_error_m",
        "pct_error",
    ]
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        return

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
            return

        header = rows[0]
        if "validation_distance_m" not in header:
            header.extend(["validation_distance_m", "abs_error_m", "pct_error"])
            for r in rows[1:]:
                while len(r) < len(header):
                    r.append("")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            logger.info("Preserved existing data and upgraded header in: %s", csv_path)
    except Exception as exc:
        logger.warning("Could not verify or migrate header for '%s': %s", csv_path, exc)


def run_live(args: argparse.Namespace) -> None:
    """Run live webcam inference with position guidance and approximate monocular range."""
    import cv2
    from ultralytics import YOLO

    model_path = validate_model_path(args.model, context="live inference")
    logger.info("Loading fine-tuned model: %s", model_path)
    model = YOLO(model_path)

    try:
        cam_source: Any = int(args.camera)
    except ValueError:
        cam_source = args.camera

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        logger.error("Could not open camera device / stream '%s'.", str(args.camera))
        logger.error("Try --camera 1 or pass a valid video file / RTSP stream path.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Range calibration configuration
    range_focal_px = None
    calibration_desc = "uncalibrated"
    if args.focal_length_px is not None:
        if args.focal_length_px <= 0:
            raise ValueError("--focal-length-px must be greater than zero")
        range_focal_px = float(args.focal_length_px)
        calibration_desc = f"fx={range_focal_px:.2f}px"
    elif args.hfov_deg is not None:
        if not (1.0 <= args.hfov_deg <= 179.0):
            raise ValueError("--hfov-deg must be between 1 and 179 degrees")
        range_focal_px = args.width / (2.0 * math.tan(math.radians(args.hfov_deg) / 2.0))
        calibration_desc = f"hfov={args.hfov_deg:.1f}deg (fx={range_focal_px:.2f}px @ {args.width}w)"

    if args.target_size_m is not None and args.target_size_m <= 0:
        raise ValueError("--target-size-m must be greater than zero")

    if args.validation_distance_m is not None and args.validation_distance_m <= 0:
        raise ValueError("--validation-distance-m must be greater than zero")

    range_log_path = os.path.abspath(args.range_log)
    os.makedirs(os.path.dirname(range_log_path) or ".", exist_ok=True)
    ensure_range_log_file(range_log_path)

    logger.info("=" * 65)
    logger.info("PerchLink CV — Live Pole Detector & Monocular Range")
    logger.info("=" * 65)
    logger.info("Camera Device: %s | Requested Resolution: %dx%d | Conf: %.2f", str(args.camera), args.width, args.height, args.conf)
    logger.info("Model Checkpoint: %s (Phase 5 frozen baseline)", model_path)
    logger.info("Runtime Range CSV Log: %s", range_log_path)

    if args.validation_distance_m is not None:
        logger.info(
            "Ground-Truth Validation Distance: %.3f m (Tape-measured) | Error logging ACTIVE",
            args.validation_distance_m
        )

    if args.target_size_m is not None and (args.hfov_deg is not None or args.focal_length_px is not None):
        logger.info(
            "Approximate Monocular Range: ENABLED | Calibration: %s | Measured Dim: %.4f m (%s axis)",
            calibration_desc, args.target_size_m, args.range_dimension
        )
        logger.warning(
            "DISCLAIMER: Monocular range is a rough pinhole geometric estimate (Z ≈ f*D/d). "
            "It is NOT safety-rated, NOT LiDAR-equivalent, and must NOT be used for autonomous control."
        )
    else:
        missing_reasons = []
        if args.hfov_deg is None and args.focal_length_px is None:
            missing_reasons.append("camera calibration (--hfov-deg or --focal-length-px)")
        if args.target_size_m is None:
            missing_reasons.append("measured target dimension (--target-dim-m)")
        logger.info(
            "Approximate Monocular Range: DISABLED (Range unavailable: missing %s)",
            " and ".join(missing_reasons)
        )

    if args.range_dimension == "width":
        logger.warning(
            "PERCH-SCENE LIMITATION: Box width corresponds to diameter ONLY for VERTICAL poles. "
            "For horizontal bars/pipes, box width is the longitudinal bar span, NOT diameter. "
            "Distance to horizontal bars CANNOT be estimated from box width."
        )

    if not args.no_display:
        logger.info("Press 'q' in the video window to quit, or Ctrl+C in terminal.")

    # Graceful shutdown handler
    shutdown_requested = False

    def _signal_handler(sig, frame):  # noqa: ANN001
        nonlocal shutdown_requested
        logger.info("Shutdown signal received — exiting gracefully.")
        shutdown_requested = True

    prev_sigint = signal.signal(signal.SIGINT, _signal_handler)
    prev_sigterm = signal.signal(signal.SIGTERM, _signal_handler)

    last_log_time = 0.0
    frame_count = 0

    try:
        while not shutdown_requested:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera.")
                break

            frame_count += 1
            if args.max_frames and frame_count >= args.max_frames:
                logger.info("Reached maximum requested frames: %d", args.max_frames)
                break

            frame_w = frame.shape[1]
            frame_h = frame.shape[0]
            frame_center_x = frame_w // 2

            # Recalculate focal length in pixels dynamically if actual frame width differs
            if args.hfov_deg is not None and args.focal_length_px is None:
                range_focal_px = frame_w / (2.0 * math.tan(math.radians(args.hfov_deg) / 2.0))
                calibration_desc = f"hfov={args.hfov_deg:.1f}deg (fx={range_focal_px:.2f}px @ {frame_w}w)"

            # YOLO inference
            results = model(
                frame,
                conf=args.conf,
                verbose=False
            )

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = model.names.get(class_id, f"cls_{class_id}")

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        measured_px = float(x2 - x1) if args.range_dimension == "width" else float(y2 - y1)
                        distance_m, range_status = estimate_monocular_range(
                            pixel_size=measured_px,
                            known_size_m=args.target_size_m,
                            focal_length_px=range_focal_px,
                            min_pixel_size=args.min_pixel_size,
                        )

                        # Determine position relative to drone camera center
                        deadzone = 100
                        if center_x < frame_center_x - deadzone:
                            position = "LEFT"
                        elif center_x > frame_center_x + deadzone:
                            position = "RIGHT"
                        else:
                            position = "CENTER"

                        # Draw box and center
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                        label_text = f"{class_name} {confidence:.2f}"
                        cv2.putText(
                            frame, label_text, (x1, max(25, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                        cv2.putText(
                            frame, f"Alignment: {position}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                        )
                        cv2.putText(
                            frame, f"Target: ({center_x}, {center_y})", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )

                        abs_error_m: Optional[float] = None
                        pct_error: Optional[float] = None
                        if args.validation_distance_m is not None:
                            val_dist_str = f"{args.validation_distance_m:.3f}"
                            if distance_m is not None:
                                abs_error_m = abs(distance_m - args.validation_distance_m)
                                pct_error = (abs_error_m / args.validation_distance_m) * 100.0
                                abs_err_str = f"{abs_error_m:.3f}"
                                pct_err_str = f"{pct_error:.1f}"
                            else:
                                abs_err_str = ""
                                pct_err_str = ""
                        else:
                            val_dist_str = ""
                            abs_err_str = ""
                            pct_err_str = ""

                        if distance_m is not None:
                            range_text = f"Approx Z: ~{distance_m:.2f} m (rough, non-safety)"
                            range_color = (0, 255, 255)
                        else:
                            range_text = f"Range: unavailable ({range_status})"
                            range_color = (0, 165, 255)

                        cv2.putText(
                            frame, range_text, (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, range_color, 2
                        )

                        if args.validation_distance_m is not None:
                            if abs_error_m is not None and pct_error is not None:
                                val_text = f"Truth: {args.validation_distance_m:.2f} m | Err: {abs_error_m:.2f} m ({pct_error:.1f}%)"
                                val_color = (0, 255, 0) if pct_error <= 20.0 else (0, 165, 255)
                            else:
                                val_text = f"Truth: {args.validation_distance_m:.2f} m (estimate unavailable)"
                                val_color = (180, 180, 180)
                            cv2.putText(
                                frame, val_text, (20, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, val_color, 2
                            )

                        # Append observation to single runtime CSV
                        with open(range_log_path, "a", newline="", encoding="utf-8") as log_file:
                            writer = csv.writer(log_file)
                            writer.writerow([
                                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                frame_count,
                                class_name,
                                f"{confidence:.4f}",
                                x1,
                                y1,
                                x2,
                                y2,
                                f"{measured_px:.1f} ({args.range_dimension})",
                                f"{args.target_size_m:.4f}" if args.target_size_m is not None else "",
                                calibration_desc,
                                f"{distance_m:.3f}" if distance_m is not None else "",
                                range_status,
                                val_dist_str,
                                abs_err_str,
                                pct_err_str,
                            ])

                        # Rate-limited logging (~1 Hz)
                        now = time.monotonic()
                        if now - last_log_time >= 1.0:
                            if distance_m is not None:
                                if args.validation_distance_m is not None and abs_error_m is not None and pct_error is not None:
                                    logger.info(
                                        "Target: %s | Conf: %.2f | Center: (%d, %d) | %s | Approx Z ≈ %.2f m | Truth: %.2f m | Err: %.2f m (%.1f%%) (rough, non-safety)",
                                        class_name, confidence, center_x, center_y, position, distance_m, args.validation_distance_m, abs_error_m, pct_error
                                    )
                                else:
                                    logger.info(
                                        "Target: %s | Conf: %.2f | Center: (%d, %d) | %s | Approx Z ≈ %.2f m (rough, non-safety)",
                                        class_name, confidence, center_x, center_y, position, distance_m
                                    )
                            else:
                                if args.validation_distance_m is not None:
                                    logger.info(
                                        "Target: %s | Conf: %.2f | Center: (%d, %d) | %s | Range: unavailable (%s) | Truth: %.2f m",
                                        class_name, confidence, center_x, center_y, position, range_status, args.validation_distance_m
                                    )
                                else:
                                    logger.info(
                                        "Target: %s | Conf: %.2f | Center: (%d, %d) | %s | Range: unavailable (%s)",
                                        class_name, confidence, center_x, center_y, position, range_status
                                    )
                            last_log_time = now

                    except Exception as exc:
                        logger.warning("Error processing detection box: %s", exc)
                        continue

            # Draw center crosshair
            cv2.line(
                frame,
                (frame_center_x, 0),
                (frame_center_x, frame_h),
                (255, 0, 0),
                1
            )

            # Frame footer disclaimer
            cv2.putText(
                frame,
                "Pinhole approx only; NOT LiDAR-equivalent / NOT safety-rated",
                (20, max(20, frame_h - 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 200, 255),
                1
            )

            if not args.no_display:
                cv2.imshow("PerchLink CV — Live Pole Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        logger.info("PerchLink CV live inference closed cleanly.")


# ==============================================================================
# Subcommand 2: PREDICT (Static image / folder inference)
# ==============================================================================

def run_predict(args: argparse.Namespace) -> None:
    """Predict poles in static images or image directories."""
    from ultralytics import YOLO

    model_path = validate_model_path(args.model, context="prediction")
    logger.info("Model: %s", model_path)

    # Collect source images
    sources: List[str] = []
    for s in args.source:
        s_abs = os.path.abspath(s)
        if os.path.isdir(s_abs):
            for fname in sorted(os.listdir(s_abs)):
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    sources.append(os.path.join(s_abs, fname))
        elif os.path.isfile(s_abs):
            if s_abs.lower().endswith(IMAGE_EXTENSIONS):
                sources.append(s_abs)
            else:
                logger.warning("File is not a supported image format: %s", s_abs)
        else:
            logger.warning("Source path not found: %s", s_abs)

    if not sources:
        logger.error("No valid image files found in specified sources.")
        sys.exit(1)

    logger.info("Found %d image(s) for prediction.", len(sources))
    logger.info("Confidence threshold: %.2f | ImgSz: %d", args.conf, args.imgsz)

    model = YOLO(model_path)
    output_project = os.path.join(SCRIPT_DIR, "runs")

    results = model.predict(
        source=sources,
        project=output_project,
        name=args.output,
        exist_ok=True,
        save=True,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )

    total_detections = sum(len(r.boxes) for r in results)
    out_dir = os.path.join(output_project, args.output)

    logger.info("==========================================")
    logger.info("PREDICTION SUMMARY")
    logger.info("==========================================")
    logger.info("Images processed   : %d", len(sources))
    logger.info("Total detections   : %d", total_detections)
    logger.info("Outputs saved to   : %s", out_dir)
    if total_detections == 0:
        logger.warning(
            "No objects detected! Consider lowering --conf (current: %.2f)",
            args.conf
        )


# ==============================================================================
# Subcommand 3: TRAIN (Model training)
# ==============================================================================

def preflight_check(data_yaml: str, base_model: str) -> None:
    """Validate data integrity and environment before starting training."""
    errors = []
    if not os.path.isfile(data_yaml):
        errors.append(f"data.yaml not found: {data_yaml}")
    if not os.path.isfile(base_model):
        errors.append(f"Base weights not found: {base_model}")

    train_img_dir = os.path.join(SCRIPT_DIR, "dataset", "images", "train")
    train_lbl_dir = os.path.join(SCRIPT_DIR, "dataset", "labels", "train")

    if os.path.isdir(train_img_dir) and os.path.isdir(train_lbl_dir):
        img_stems = {
            os.path.splitext(f)[0] for f in os.listdir(train_img_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        }
        lbl_stems = {
            os.path.splitext(f)[0] for f in os.listdir(train_lbl_dir)
            if f.endswith(".txt")
        }

        imgs_without_labels = img_stems - lbl_stems
        labels_without_imgs = lbl_stems - img_stems
        empty_labels = [
            f for f in os.listdir(train_lbl_dir)
            if f.endswith(".txt") and os.path.getsize(os.path.join(train_lbl_dir, f)) == 0
        ]

        logger.info("[Pre-flight] Train images: %d | Train labels: %d", len(img_stems), len(lbl_stems))
        if imgs_without_labels:
            logger.warning("[Pre-flight] %d images without labels (treated as negative background)", len(imgs_without_labels))
        if labels_without_imgs:
            logger.warning("[Pre-flight] %d label files without corresponding images", len(labels_without_imgs))
        if empty_labels:
            logger.info("[Pre-flight] %d hard-negative images (empty labels)", len(empty_labels))
    else:
        errors.append("Dataset train images or labels directory missing.")

    if errors:
        for err in errors:
            logger.error("[Pre-flight Error] %s", err)
        raise RuntimeError("Pre-flight checks failed. Cannot proceed with training.")


def run_train(args: argparse.Namespace) -> Any:
    """Run YOLO training with drone-specific augmentation hyperparameters."""
    from ultralytics import YOLO

    data_path = os.path.abspath(args.data)
    model_path = validate_model_path(args.model, context="base training")

    preflight_check(data_path, model_path)

    logger.info("Initializing YOLO base model: %s", model_path)
    model = YOLO(model_path)

    logger.info("Beginning training experiment '%s'...", args.name)
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        pretrained=True,
        freeze=args.freeze,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.5,
        translate=0.1,
        scale=args.scale,
        shear=0.0,
        fliplr=args.fliplr,
        degrees=args.degrees,
        perspective=0.0002,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        auto_augment="none",
        device=args.device,
        workers=args.workers,
        amp=True,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        single_cls=True,
        seed=args.seed,
        deterministic=True,
        verbose=True,
        plots=True,
        project=os.path.abspath(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        save_period=args.save_period,
    )
    logger.info("Training experiment '%s' completed successfully.", args.name)
    return results


# ==============================================================================
# Subcommand 4: EVALUATE (Validation & Test Evaluation + Threshold Sweep)
# ==============================================================================

def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate fine-tuned checkpoint on validation or test split."""
    from ultralytics import YOLO

    model_path = validate_model_path(args.model, context="evaluation")
    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    logger.info("Evaluating model: %s on split '%s'", model_path, args.split)
    model = YOLO(model_path)

    # Step 1: Standard benchmark mAP (untruncated PR curve down to conf=0.001)
    logger.info("Computing standard COCO/YOLO benchmark mAP (conf=0.001, iou=0.7)...")
    std_metrics = model.val(
        data=data_path,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
        plots=False
    )
    std_map50 = float(std_metrics.box.map50)
    std_map50_95 = float(std_metrics.box.map)

    if args.sweep:
        logger.info("==================================================")
        logger.info("OPERATING THRESHOLD SWEEP (Split: %s)", args.split)
        logger.info("==================================================")
        print(f"{'conf':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
        print("-" * 40)

        thresholds = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
        best_f1 = 0.0
        best_conf = args.conf

        for conf in thresholds:
            m = model.val(
                data=data_path,
                split=args.split,
                conf=conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                plots=False
            )
            p = float(m.box.p[0]) if len(m.box.p) else 0.0
            r = float(m.box.r[0]) if len(m.box.r) else 0.0
            f1 = 2 * p * r / (p + r + 1e-9)

            marker = " *" if f1 > best_f1 else ""
            print(f"{conf:>6.2f}  {p:>10.4f}  {r:>8.4f}  {f1:>8.4f}{marker}")
            if f1 > best_f1:
                best_f1 = f1
                best_conf = conf

        logger.info("Sweep complete. Optimal F1 threshold: conf=%.2f (F1=%.4f)", best_conf, best_f1)
        eval_conf = best_conf
    else:
        eval_conf = args.conf

    # Step 2: Operational evaluation at the chosen operating threshold
    logger.info("Evaluating operational metrics at operating threshold conf=%.2f, iou=%.2f...", eval_conf, args.iou)
    op_metrics = model.val(
        data=data_path,
        split=args.split,
        conf=eval_conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        plots=True
    )

    prec = float(op_metrics.box.p[0]) if len(op_metrics.box.p) else 0.0
    rec = float(op_metrics.box.r[0]) if len(op_metrics.box.r) else 0.0
    f1_val = 2 * prec * rec / (prec + rec + 1e-9)

    results_dict = {
        "model": model_path,
        "split": args.split,
        "standard_benchmark": {
            "mAP50": round(std_map50, 4),
            "mAP50-95": round(std_map50_95, 4),
            "eval_protocol": "untruncated PR curve (conf=0.001, iou=0.7)"
        },
        "operating_point": {
            "operating_conf": eval_conf,
            "iou_threshold": args.iou,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1_val, 4),
        }
    }

    logger.info("==========================================")
    logger.info("EVALUATION RESULTS (%s split)", args.split.upper())
    logger.info("==========================================")
    print(json.dumps(results_dict, indent=2))



# ==============================================================================
# Subcommand 5: EXPORT (Export to ONNX / NCNN)
# ==============================================================================

def run_export(args: argparse.Namespace) -> None:
    """Export model to ONNX or other edge-deployable formats."""
    from ultralytics import YOLO

    model_path = validate_model_path(args.model, context="export")
    logger.info("Exporting model: %s to format: %s", model_path, args.format)
    model = YOLO(model_path)

    exported_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        simplify=args.simplify,
        dynamic=args.dynamic,
        opset=args.opset,
        half=args.half,
    )
    logger.info("Export complete. Artifact saved to: %s", exported_path)


# ==============================================================================
# Subcommand 6: CONVERT (HOTOSM Parquet to YOLO)
# ==============================================================================

def run_convert(args: argparse.Namespace) -> None:
    """Convert raw HOTOSM parquet files into YOLO images and label splits.
    
    CRITICAL: Destructive operations require explicit opt-in confirmation!
    """
    import pandas as pd
    from PIL import Image, ImageFile
    from io import BytesIO
    from tqdm import tqdm

    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.dest)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source HOTOSM directory not found: {source_dir}")

    # Safety: Clean previous conversion only with explicit opt-in
    if args.clean:
        if not args.confirm:
            err_msg = (
                "Destructive --clean requested without --confirm flag.\n"
                "Aborting conversion to prevent accidental data loss.\n"
                "Pass --confirm to permit removing existing images/labels folders."
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        logger.warning("Cleaning existing images/labels in %s...", dest_dir)
        for folder in [os.path.join(dest_dir, "images"), os.path.join(dest_dir, "labels")]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                logger.info("Removed: %s", folder)

    # Bomb protection
    Image.MAX_IMAGE_PIXELS = 89_478_485
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, "labels", split), exist_ok=True)

        parquet_dir = os.path.join(source_dir, "data", split)
        if not os.path.isdir(parquet_dir):
            logger.warning("No data directory found for split '%s' at %s", split, parquet_dir)
            continue

        parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        logger.info("Processing split '%s' (%d parquet files)...", split, len(parquet_files))

        total_images = 0
        total_poles = 0
        images_with_poles = 0

        for pq_file in parquet_files:
            df = pd.read_parquet(pq_file)
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Converting {split}"):
                image_data = row["image"]
                if isinstance(image_data, dict):
                    image_bytes = image_data.get("bytes") or image_data.get("data")
                elif isinstance(image_data, bytes):
                    image_bytes = image_data
                else:
                    continue

                if not image_bytes:
                    continue

                try:
                    buf = BytesIO(image_bytes)
                    probe = Image.open(buf)
                    probe.verify()
                    buf.seek(0)
                    img = Image.open(buf).convert("RGB")
                except Exception as e:
                    logger.warning("Skipping invalid image at %s: %s", idx, e)
                    continue

                width, height = img.size
                objects = row.get("objects", {})
                bboxes = objects.get("bbox", [])
                categories = objects.get("category", [])

                yolo_lines = []
                for bbox, cat in zip(bboxes, categories):
                    if int(cat) != 0:
                        continue  # Only keep class 0 (pole)
                    x, y, w, h = map(float, bbox)
                    if w <= 0 or h <= 0:
                        continue
                    if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
                        continue

                    # Normalized YOLO coordinates
                    x_c = max(0.0, min(1.0, (x + w / 2.0) / width))
                    y_c = max(0.0, min(1.0, (y + h / 2.0) / height))
                    w_n = max(0.0, min(1.0, w / width))
                    h_n = max(0.0, min(1.0, h / height))

                    yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
                    total_poles += 1

                img_name = f"{split}_{total_images:07d}.jpg"
                img_path = os.path.join(dest_dir, "images", split, img_name)
                lbl_path = os.path.join(dest_dir, "labels", split, f"{split}_{total_images:07d}.txt")

                img.save(img_path, quality=95)
                if yolo_lines:
                    with open(lbl_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(yolo_lines))
                    images_with_poles += 1

                total_images += 1

        logger.info(
            "Split %s complete: %d images saved, %d with poles, %d pole bboxes.",
            split, total_images, images_with_poles, total_poles
        )

    # Generate data.yaml
    yaml_path = os.path.join(dest_dir, "data.yaml")
    yaml_content = (
        f"path: {pathlib.Path(dest_dir).as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: pole\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    logger.info("Generated data.yaml at: %s", yaml_path)


# ==============================================================================
# Subcommand 7: AUDIT (Dataset & Annotation Quality Audit)
# ==============================================================================

def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Audit images, labels, bounds, aspect ratios, and duplicate boxes."""
    dataset_dir = os.path.abspath(args.dataset)
    splits = ["train", "val", "test"]

    stats = {
        "images": {"train": 0, "val": 0, "test": 0, "total": 0},
        "positive_images": {"train": 0, "val": 0, "test": 0, "total": 0},
        "empty_labels": {"train": 0, "val": 0, "test": 0, "total": 0},
        "total_boxes": {"train": 0, "val": 0, "test": 0, "total": 0},
        "missing_labels": [],
        "invalid_annotations": [],
        "too_large_boxes": [],
        "too_small_boxes": [],
        "duplicate_boxes": [],
    }

    image_hashes = defaultdict(list)

    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

        if not os.path.exists(img_dir):
            continue

        imgs = []
        for ext in IMAGE_EXTENSIONS:
            imgs.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))
        imgs.sort()

        stats["images"][split] = len(imgs)
        stats["images"]["total"] += len(imgs)

        for img_path in imgs:
            # MD5 hash for image deduplication check
            with open(img_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
                image_hashes[h].append(img_path)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")

            if not os.path.exists(lbl_path):
                stats["missing_labels"].append(img_path)
                continue

            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) == 0:
                stats["empty_labels"][split] += 1
                stats["empty_labels"]["total"] += 1
                continue

            stats["positive_images"][split] += 1
            stats["positive_images"]["total"] += 1

            boxes = []
            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    stats["invalid_annotations"].append((lbl_path, i, "Not 5 values"))
                    continue
                try:
                    c = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    if c != 0:
                        stats["invalid_annotations"].append((lbl_path, i, f"Invalid class ID: {c}"))
                    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        stats["invalid_annotations"].append((lbl_path, i, f"Out of bounds: {parts[1:]}"))
                        continue

                    box = (xc, yc, w, h)
                    boxes.append(box)
                    stats["total_boxes"][split] += 1
                    stats["total_boxes"]["total"] += 1

                    if w > args.max_size or h > args.max_size:
                        stats["too_large_boxes"].append((lbl_path, i, box))
                    if w < args.min_size or h < args.min_size:
                        stats["too_small_boxes"].append((lbl_path, i, box))

                except ValueError:
                    stats["invalid_annotations"].append((lbl_path, i, "Float conversion error"))

            # Duplicate box check within same image
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = compute_iou(boxes[i], boxes[j])
                    if iou > args.iou_thresh:
                        stats["duplicate_boxes"].append((lbl_path, i, j, iou, boxes[i], boxes[j]))

    duplicate_images = {h: paths for h, paths in image_hashes.items() if len(paths) > 1}

    logger.info("==========================================")
    logger.info("PERCHLINK CV — DATASET AUDIT REPORT")
    logger.info("==========================================")
    print(f"Dataset root               : {dataset_dir}")
    print(f"Total images               : {stats['images']['total']}")
    print(f"  - Train                  : {stats['images']['train']}")
    print(f"  - Val                    : {stats['images']['val']}")
    print(f"  - Test                   : {stats['images']['test']}")
    print(f"Positive images            : {stats['positive_images']['total']}")
    print(f"Empty labels (Background)  : {stats['empty_labels']['total']} "
          f"({(stats['empty_labels']['total'] / max(1, stats['images']['total']) * 100):.1f}%)")
    print(f"Total bounding boxes       : {stats['total_boxes']['total']}")
    print(f"  - Train boxes            : {stats['total_boxes']['train']}")
    print(f"  - Val boxes              : {stats['total_boxes']['val']}")
    print(f"  - Test boxes             : {stats['total_boxes']['test']}")
    print(f"Missing label files        : {len(stats['missing_labels'])}")
    print(f"Invalid annotations        : {len(stats['invalid_annotations'])}")
    print(f"Too large boxes (> {args.max_size*100:.0f}%)   : {len(stats['too_large_boxes'])}")
    print(f"Too small boxes (< {args.min_size*100:.0f}%)   : {len(stats['too_small_boxes'])}")
    print(f"Duplicate boxes (IoU>{args.iou_thresh}) : {len(stats['duplicate_boxes'])}")
    print(f"Identical duplicate images : {len(duplicate_images)}")

    if stats["missing_labels"]:
        print("\nSample missing labels:")
        for item in stats["missing_labels"][:3]:
            print(f"  {item}")

    if stats["invalid_annotations"]:
        print("\nSample invalid annotations:")
        for item in stats["invalid_annotations"][:3]:
            print(f"  {item}")

    if stats["duplicate_boxes"]:
        print("\nSample duplicate bounding boxes:")
        for item in stats["duplicate_boxes"][:3]:
            print(f"  File: {item[0]} | Boxes: #{item[1]}, #{item[2]} | IoU: {item[3]:.3f}")

    return stats


# ==============================================================================
# Subcommand 8: PREVIEW (Dataset Ground-Truth Inspection)
# ==============================================================================

def run_preview(args: argparse.Namespace) -> None:
    """Preview dataset ground truth bounding boxes."""
    import cv2

    dataset_dir = os.path.abspath(args.dataset)
    image_dir = os.path.join(dataset_dir, "images", args.split)
    label_dir = os.path.join(dataset_dir, "labels", args.split)

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    random.seed(args.seed)
    random.shuffle(images)

    logger.info("Previewing %s split | Seed: %d | Count: %d", args.split, args.seed, args.count)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info("Saving preview images to: %s", args.save_dir)

    shown = 0
    for filename in images:
        img_path = os.path.join(image_dir, filename)
        lbl_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")

        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            continue

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id, xc, yc, bw, bh = map(float, parts)
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                continue

            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, "POLE", (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        if args.save_dir:
            out_file = os.path.join(args.save_dir, f"preview_{filename}")
            cv2.imwrite(out_file, image)
        elif not args.no_gui:
            cv2.imshow("PerchLink CV — Dataset Ground Truth Preview", image)
            logger.info("Showing %s [SPACE/Any Key: next, Q: quit]", filename)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

        shown += 1
        if shown >= args.count:
            break

    if not args.save_dir and not args.no_gui:
        cv2.destroyAllWindows()
    logger.info("Preview finished. Displayed/saved %d image(s).", shown)


# ==============================================================================
# Subcommand 9: CURATE (Opt-in Background Ratio Curation)
# ==============================================================================

def run_curate(args: argparse.Namespace) -> None:
    """Curate dataset by trimming excess background images.
    
    CRITICAL: Destructive operations require explicit opt-in via --apply!
    """
    dataset_dir = os.path.abspath(args.dataset)
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    random.seed(args.seed)

    logger.info("==========================================")
    logger.info("DATASET CURATION (Target background ratio: %.2f)", args.bg_ratio)
    logger.info("==========================================")

    if not args.apply:
        logger.warning("[DRY-RUN MODE] No files will be deleted. Pass --apply to perform deletions.")

    for split in splits:
        labels_dir = os.path.join(dataset_dir, "labels", split)
        images_dir = os.path.join(dataset_dir, "images", split)

        if not os.path.isdir(labels_dir) or not os.path.isdir(images_dir):
            logger.warning("Split directories for '%s' not found. Skipping.", split)
            continue

        all_labels = glob.glob(os.path.join(labels_dir, "*.txt"))
        empty_labels = []
        positive_labels = []

        for lbl in all_labels:
            if os.path.getsize(lbl) == 0:
                empty_labels.append(lbl)
            else:
                positive_labels.append(lbl)

        target_bg_count = int(len(positive_labels) * args.bg_ratio)
        excess_count = max(0, len(empty_labels) - target_bg_count)

        logger.info(
            "Split '%s': %d positives, %d backgrounds | Target backgrounds: %d | Excess: %d",
            split, len(positive_labels), len(empty_labels), target_bg_count, excess_count
        )

        if excess_count > 0:
            random.shuffle(empty_labels)
            to_delete = empty_labels[target_bg_count:]
            if args.apply:
                deleted_images = 0
                deleted_labels = 0
                for lbl in to_delete:
                    base = os.path.splitext(os.path.basename(lbl))[0]
                    # Check for image with any supported extension
                    for ext in IMAGE_EXTENSIONS:
                        img_path = os.path.join(images_dir, base + ext)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            deleted_images += 1
                            break
                    if os.path.exists(lbl):
                        os.remove(lbl)
                        deleted_labels += 1
                logger.info("  [APPLIED] Deleted %d images and %d labels.", deleted_images, deleted_labels)
            else:
                logger.info("  [DRY-RUN] Would delete %d excess background pairs.", len(to_delete))
        else:
            logger.info("  No excess background images to delete.")


# ==============================================================================
# Subcommand 10: SPLIT (Opt-in Dataset Re-splitting)
# ==============================================================================

def run_split(args: argparse.Namespace) -> None:
    """Re-split dataset into train/val/test partitions.
    
    CRITICAL: Destructive / moving operations require explicit opt-in via --apply!
    """
    dataset_dir = os.path.abspath(args.dataset)
    splits = ["train", "val", "test"]

    all_pairs: List[Tuple[str, str]] = []
    for split in splits:
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        imgs = []
        for ext in IMAGE_EXTENSIONS:
            imgs.extend(glob.glob(os.path.join(img_dir, f"*{ext}")))

        for img in imgs:
            base = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(lbl_dir, base + ".txt")
            if os.path.exists(lbl):
                all_pairs.append((img, lbl))
            else:
                logger.warning("No label found for image: %s", img)

    total = len(all_pairs)
    if total == 0:
        logger.error("No valid image-label pairs found in dataset: %s", dataset_dir)
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(all_pairs)

    train_end = int(total * args.train_ratio)
    val_end = train_end + int(total * args.val_ratio)

    split_data = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:]
    }

    logger.info("==========================================")
    logger.info("DATASET RE-SPLIT CONFIGURATION")
    logger.info("==========================================")
    logger.info("Total pairs: %d", total)
    logger.info("Target distribution: Train=%d (%.1f%%), Val=%d (%.1f%%), Test=%d (%.1f%%)",
                len(split_data["train"]), len(split_data["train"])/total*100,
                len(split_data["val"]), len(split_data["val"])/total*100,
                len(split_data["test"]), len(split_data["test"])/total*100)

    if not args.apply:
        logger.warning("[DRY-RUN MODE] No files were moved. Pass --apply to execute re-splitting.")
        return

    logger.warning("Executing dataset re-split via staging directory...")
    tmp_dir = os.path.join(dataset_dir, "_tmp_split_staging")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # Step 1: Move pairs to staging
        for split_name, pairs in split_data.items():
            os.makedirs(os.path.join(tmp_dir, "images", split_name), exist_ok=True)
            os.makedirs(os.path.join(tmp_dir, "labels", split_name), exist_ok=True)
            for img_path, lbl_path in pairs:
                shutil.move(img_path, os.path.join(tmp_dir, "images", split_name, os.path.basename(img_path)))
                shutil.move(lbl_path, os.path.join(tmp_dir, "labels", split_name, os.path.basename(lbl_path)))

        # Step 2: Clear old split directories
        for split_name in splits:
            for sub in ["images", "labels"]:
                d = os.path.join(dataset_dir, sub, split_name)
                if os.path.exists(d):
                    for f in os.listdir(d):
                        os.remove(os.path.join(d, f))

        # Step 3: Move from staging to final location
        for split_name in splits:
            for sub in ["images", "labels"]:
                src = os.path.join(tmp_dir, sub, split_name)
                dst = os.path.join(dataset_dir, sub, split_name)
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(src):
                    shutil.move(os.path.join(src, f), os.path.join(dst, f))

        logger.info("Dataset re-splitting completed successfully.")
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ==============================================================================
# CLI Argument Parser Construction
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build unified command line interface for PerchLink CV."""
    parser = argparse.ArgumentParser(
        prog="perchlink_cv",
        description="PerchLink Computer Vision Pipeline — Unified CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. LIVE
    p_live = subparsers.add_parser("live", help="Run live camera pole detection with optional monocular range")
    p_live.add_argument("--model", type=str, default=DEFAULT_BEST_CHECKPOINT,
                        help="Path to fine-tuned weights (strict, no COCO fallback)")
    p_live.add_argument("--camera", type=str, default="0",
                        help="Camera device index (e.g. 0) or video file / stream path")
    p_live.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (frozen Phase 5 default: 0.35)")
    p_live.add_argument("--width", type=int, default=1280, help="Camera frame width")
    p_live.add_argument("--height", type=int, default=720, help="Camera frame height")
    p_live.add_argument("--no-display", action="store_true", help="Run headless without OpenCV window")
    p_live.add_argument("--max-frames", type=int, default=None, help="Exit after N frames")
    p_live.add_argument("--target-size-m", "--target-dim-m", "--target-dim", type=float, default=None,
                        dest="target_size_m",
                        help="User-entered measured target dimension in metres (e.g. 0.20 for a 20cm pole diameter). "
                             "Required for range estimation. Never guessed from model or class.")
    p_live.add_argument("--range-dimension", "--target-dim-axis", "--dim-axis", choices=["width", "height"], default="width",
                        dest="range_dimension",
                        help="Bounding-box pixel dimension corresponding to the measured target dimension ('width' or 'height'). "
                             "PERCH-SCENE LIMITATION: 'width' corresponds to diameter ONLY for vertical poles. For horizontal bars, "
                             "box width spans the longitudinal bar length and CANNOT be used to estimate range from diameter.")
    p_live.add_argument("--hfov-deg", "--hfov", "--fov", type=float, default=None,
                        dest="hfov_deg",
                        help="Camera horizontal field of view in degrees (1-179 deg). Used to calculate focal length in pixels.")
    p_live.add_argument("--focal-length-px", "--focal-length", "--fx", type=float, default=None,
                        dest="focal_length_px",
                        help="Calibrated camera focal length in pixels. Direct alternative to horizontal FOV.")
    p_live.add_argument("--min-pixel-size", type=float, default=8.0,
                        dest="min_pixel_size",
                        help="Minimum reliable bounding-box pixel dimension for range estimation (default: 8.0 px)")
    p_live.add_argument("--range-log", type=str,
                        default=os.path.join(SCRIPT_DIR, "runs", "perchlink_range_estimates.csv"),
                        help="Single append-only CSV file for runtime range observations (default: runs/perchlink_range_estimates.csv)")
    p_live.add_argument("--validation-distance-m", "--validation-dist-m", "--val-dist-m", "--val-distance-m", "--truth-dist-m",
                        type=float, default=None,
                        dest="validation_distance_m",
                        help="Optional tape-measured ground-truth camera-to-target distance in metres for bench validation. "
                             "When provided, logs truth distance, absolute error (m), and percentage error (%%) to CSV.")

    # 2. PREDICT
    p_pred = subparsers.add_parser("predict", help="Predict poles in static images")
    p_pred.add_argument("source", nargs="+", help="Image file(s) or directories")
    p_pred.add_argument("--model", type=str, default=DEFAULT_BEST_CHECKPOINT,
                        help="Path to fine-tuned weights")
    p_pred.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_pred.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    p_pred.add_argument("--output", type=str, default="predictions", help="Output folder name in runs/")
    p_pred.add_argument("--device", type=str, default="0", help="CUDA device or cpu")
    p_pred.add_argument("--save-txt", action="store_true", default=True, help="Save YOLO labels")
    p_pred.add_argument("--save-conf", action="store_true", default=True, help="Save confidences")

    # 3. TRAIN
    p_train = subparsers.add_parser("train", help="Train YOLO model on pole dataset")
    p_train.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="data.yaml path")
    p_train.add_argument("--model", type=str, default=os.path.join(SCRIPT_DIR, "yolo11n.pt"),
                         help="Base model checkpoint")
    p_train.add_argument("--epochs", type=int, default=100, help="Training epochs")
    p_train.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    p_train.add_argument("--imgsz", type=int, default=1024, help="Input resolution")
    p_train.add_argument("--batch", type=int, default=8, help="Batch size")
    p_train.add_argument("--freeze", type=int, default=10, help="Backbone layers to freeze")
    p_train.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    p_train.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    p_train.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    p_train.add_argument("--momentum", type=float, default=0.937, help="Momentum")
    p_train.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    p_train.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")
    p_train.add_argument("--box", type=float, default=7.5, help="Box loss gain")
    p_train.add_argument("--cls", type=float, default=0.3, help="Class loss gain")
    p_train.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain")
    p_train.add_argument("--degrees", type=float, default=10.0, help="Rotation degrees augmentation")
    p_train.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability (keep 0.0)")
    p_train.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability")
    p_train.add_argument("--scale", type=float, default=0.6, help="Scale augmentation")
    p_train.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation")
    p_train.add_argument("--mixup", type=float, default=0.1, help="Mixup augmentation")
    p_train.add_argument("--copy_paste", type=float, default=0.1, help="Copy-paste augmentation")
    p_train.add_argument("--erasing", type=float, default=0.3, help="Random erasing augmentation")
    p_train.add_argument("--close_mosaic", type=int, default=10, help="Close mosaic epochs before end")
    p_train.add_argument("--device", type=str, default="0", help="CUDA device or cpu")
    p_train.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    p_train.add_argument("--cache", action="store_true", default=True, help="Cache dataset in RAM")
    p_train.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    p_train.add_argument("--project", type=str, default=os.path.join(SCRIPT_DIR, "runs"), help="Project dir")
    p_train.add_argument("--name", type=str, default="experiment_baseline", help="Experiment name")
    p_train.add_argument("--exist-ok", action="store_true", default=False, help="Overwrite run folder")
    p_train.add_argument("--save-period", type=int, default=10, help="Checkpoint period")

    # 4. EVALUATE
    p_eval = subparsers.add_parser("evaluate", aliases=["val"], help="Evaluate model metrics")
    p_eval.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="data.yaml path")
    p_eval.add_argument("--model", type=str, default=DEFAULT_BEST_CHECKPOINT,
                        help="Checkpoint path to evaluate")
    p_eval.add_argument("--split", type=str, default="val", choices=["val", "test", "train"],
                        help="Split to evaluate on")
    p_eval.add_argument("--imgsz", type=int, default=1024, help="Resolution")
    p_eval.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_eval.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p_eval.add_argument("--device", type=str, default="0", help="CUDA device or cpu")
    p_eval.add_argument("--sweep", action="store_true", help="Run confidence threshold sweep")

    # 5. EXPORT
    p_exp = subparsers.add_parser("export", help="Export model to ONNX or edge formats")
    p_exp.add_argument("--model", type=str, default=DEFAULT_BEST_CHECKPOINT, help="Model weights")
    p_exp.add_argument("--format", type=str, default="onnx", choices=["onnx", "ncnn", "tflite", "torchscript"],
                       help="Target format")
    p_exp.add_argument("--imgsz", type=int, default=320, help="Inference input size")
    p_exp.add_argument("--half", action="store_true", default=False, help="FP16 half precision")
    p_exp.add_argument("--opset", type=int, default=12, help="ONNX opset")
    p_exp.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph")
    p_exp.add_argument("--dynamic", action="store_true", default=False, help="Dynamic axes")

    # 6. CONVERT
    p_conv = subparsers.add_parser("convert", help="Convert raw HOTOSM parquet to YOLO")
    p_conv.add_argument("--source", type=str, default=os.path.join(SCRIPT_DIR, "dataset", "hotosm"),
                        help="Source folder containing data/")
    p_conv.add_argument("--dest", type=str, default=os.path.join(SCRIPT_DIR, "dataset"),
                        help="Destination dataset directory")
    p_conv.add_argument("--clean", action="store_true", help="Clean destination before conversion (opt-in)")
    p_conv.add_argument("--confirm", action="store_true", help="Explicit confirmation for destructive clean")

    # 7. AUDIT
    p_aud = subparsers.add_parser("audit", help="Audit dataset annotations and consistency")
    p_aud.add_argument("--dataset", type=str, default=os.path.join(SCRIPT_DIR, "dataset"),
                       help="Dataset directory")
    p_aud.add_argument("--iou-thresh", type=float, default=0.85, help="IoU duplicate threshold")
    p_aud.add_argument("--max-size", type=float, default=0.95, help="Oversized box threshold")
    p_aud.add_argument("--min-size", type=float, default=0.01, help="Undersized box threshold")

    # 8. PREVIEW
    p_prev = subparsers.add_parser("preview", help="Preview dataset images and labels")
    p_prev.add_argument("--dataset", type=str, default=os.path.join(SCRIPT_DIR, "dataset"),
                        help="Dataset directory")
    p_prev.add_argument("--split", type=str, default="train", choices=["train", "val", "test"],
                        help="Split to preview")
    p_prev.add_argument("--count", type=int, default=10, help="Max images to show")
    p_prev.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p_prev.add_argument("--save-dir", type=str, default=None, help="Save preview images here instead of GUI")
    p_prev.add_argument("--no-gui", action="store_true", help="Do not show OpenCV GUI window")

    # 9. CURATE
    p_cur = subparsers.add_parser("curate", help="Balance background empty labels (opt-in)")
    p_cur.add_argument("--dataset", type=str, default=os.path.join(SCRIPT_DIR, "dataset"),
                       help="Dataset directory")
    p_cur.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"],
                       help="Split to curate")
    p_cur.add_argument("--bg-ratio", type=float, default=0.10, help="Target background to positive ratio")
    p_cur.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p_cur.add_argument("--apply", action="store_true", help="Explicit opt-in to apply deletions (default: dry-run)")

    # 10. SPLIT
    p_spl = subparsers.add_parser("split", aliases=["fix-splits"], help="Re-split dataset (opt-in)")
    p_spl.add_argument("--dataset", type=str, default=os.path.join(SCRIPT_DIR, "dataset"),
                       help="Dataset directory")
    p_spl.add_argument("--train-ratio", type=float, default=0.80, help="Train ratio")
    p_spl.add_argument("--val-ratio", type=float, default=0.10, help="Validation ratio")
    p_spl.add_argument("--test-ratio", type=float, default=0.10, help="Test ratio")
    p_spl.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    p_spl.add_argument("--apply", action="store_true", help="Explicit opt-in to apply split changes (default: dry-run)")

    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    command_handlers = {
        "live": run_live,
        "predict": run_predict,
        "train": run_train,
        "evaluate": run_evaluate,
        "val": run_evaluate,
        "export": run_export,
        "convert": run_convert,
        "audit": run_audit,
        "preview": run_preview,
        "curate": run_curate,
        "split": run_split,
        "fix-splits": run_split,
    }

    handler = command_handlers.get(args.subcommand)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
