---
license: cc-by-sa-4.0
task_categories:
  - object-detection
language:
  - en
pretty_name: Street-level Poles & Towers
size_categories:
  - 1K<n<10K
tags:
  - object-detection
  - yolo
  - streetlevel
  - panoramax
  - openstreetmap
  - poles
  - towers
  - infrastructure
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train/*.parquet
      - split: validation
        path: data/val/*.parquet
      - split: test
        path: data/test/*.parquet
---

# Street-level Poles & Towers

Object detection dataset of utility **poles** and **towers** in street-level
imagery. Curated by **UC Berkeley** as part of the HOT-OSM YOLO experiments.
This release packages the working set in three compatible layouts:

- **HuggingFace parquet** (default loader): images embedded as bytes with
  COCO-style boxes (`x, y, w, h` in pixels), class labels, and per-image
  source metadata. Loadable with `datasets.load_dataset(...)`.
- **COCO JSON** under `coco/instances_{train,val,test}.json`.
- **YOLO** dataset description in `data.yaml` (Ultralytics-compatible).

## Classes

| id | name  |
|----|-------|
| 0  | pole  |
| 1  | tower |

## Splits

| split | images | positive | negative | boxes | pole boxes | tower boxes |
|-------|-------:|---------:|---------:|------:|-----------:|------------:|
| train | 2,001  | 227      | 1,774    | 338   | 199        | 139         |
| val   | 996    | 501      | 495      | 816   | 505        | 311         |
| test  | 503    | 256      | 247      | 409   | 273        | 136         |
| total | 3,500  | 984      | 2,516    | 1,563 | 977        | 586         |

The train split is dominated by negative (no-object) samples while val and
test are roughly balanced; users training detectors may want to resample.

## Geographic coverage

1,051 negative samples carry per-image latitude / longitude / Panoramax id /
source URL sourced from [panoramax.openstreetmap.fr](https://panoramax.openstreetmap.fr).
Coverage spans Western Europe and East Asia (lat 24.8 to 53.1, lon -4.6 to
121.5). Positive samples encode their origin only as a filename prefix
(`bryan_*`, `shanghai_*`, `oakland_*`, `berkeley_*`).

Artifacts shipped with the repo: `artifacts/geo_map.html` (interactive map),
`artifacts/geo_map.png` (static scatter), `artifacts/split_class_counts.png`,
`artifacts/image_counts.png`, `artifacts/boxes_per_image.png`,
`artifacts/source_distribution.png`.

## Source provenance

| source prefix | images |
|---------------|-------:|
| bryan         | 3,452  |
| roboflow      |    42  |
| shanghai      |     5  |
| oakland       |     1  |

## Parquet schema

```python
{
  "image":       Image(decode=True),                          # JPEG bytes
  "image_id":    Value("string"),
  "width":       Value("int32"),
  "height":      Value("int32"),
  "objects": {
    "bbox":      Sequence(Sequence(Value("float32"), length=4)),   # [x, y, w, h] in pixels
    "category":  Sequence(ClassLabel(names=["pole", "tower"])),
  },
  "source":         Value("string"),
  "panoramax_id":   Value("string"),
  "source_url":     Value("string"),
  "lat":            Value("float64"),
  "lon":            Value("float64"),
}
```

## Loading

```python
from datasets import load_dataset

ds = load_dataset("kshitijrajsharma/streetlevel-poles")
ds["train"][0]["image"]            # PIL.Image
ds["train"][0]["objects"]          # {"bbox": [...], "category": [...]}
```

For Ultralytics / YOLO, use the shipped `data.yaml` and image folders
directly, or convert from the parquet shards.

## Baseline

A YOLOv11n baseline trained for 10 epochs at `imgsz=640`:

| metric          | value |
|-----------------|------:|
| mAP@50          | 0.332 |
| mAP@50-95       | 0.120 |
| precision       | 0.455 |
| recall          | 0.345 |

See `artifacts/baseline_results.png`,
`artifacts/baseline_confusion_matrix.png`, and
`artifacts/baseline_pr_curve.png`.

Trained weights and an interactive demo:

- Model (YOLOv8s, 50 epochs - original project deliverable): <https://huggingface.co/hotosm/streetlevel-poles-yolov8s>
- Model (YOLOv11n, 10-epoch sanity-check baseline): <https://huggingface.co/kshitijrajsharma/streetlevel-poles-yolo11n>
- Demo Space (runs the yolo11n baseline): <https://huggingface.co/spaces/kshitijrajsharma/streetlevel-poles-detector>

## License

Released under **CC-BY-SA-4.0**. Negative samples derive from Panoramax /
OpenStreetMap-sourced imagery; share-alike compatibility is preserved.
