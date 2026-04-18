# YOLO-EOV

This repository accompanies an **undergraduate Final Year Project (FYP)** on **open-vocabulary object detection**. It provides a lightweight, reproducible codebase built around **YOLOE-11L** for training, evaluation, and deployment of open-vocabulary detectors.

## Author

- PENG, Minqi; SID: 1155191548
- ZHU, Keyu; SID: 1155191834

## Objectives

The implementation targets the following properties:

- Model size within approximately **20M–80M** parameters  
- Support for **COCO** and **LVIS** benchmarks  
- **Open-vocabulary inference** via arbitrary text class prompts  
- End-to-end workflows for **training**, **standard AP/AR evaluation**, **throughput (FPS) measurement**, and **ONNX export**

**Default configuration:** `yoloe-11l-seg.pt` with `yoloe-11l.yaml`. This pairing offers a practical balance among latency, accuracy, and open-vocabulary capability within the lightweight regime.

---

## 1. Repository layout

```text
YOLO-EOV/
├── README.md
├── requirements.txt
├── configs/
│   ├── coco_ovd.template.yaml
│   └── lvis_ovd.template.yaml
├── ovd/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluation.py
│   └── utils.py
├── scripts/
│   ├── benchmark.py
│   ├── benchmark.sh
│   ├── eval_coco.sh
│   ├── eval_lvis.sh
│   ├── export.py
│   ├── export_onnx.sh
│   ├── model_info.py
│   ├── predict.py
│   ├── train.py
│   ├── train_coco.sh
│   ├── train_lvis.sh
│   └── val.py
└── tools/
    ├── convert_coco_to_yolo_det.py
    ├── convert_lvis_to_yolo_det.py
    └── make_dataset_yaml.py
```

---

## 2. Environment setup

**Recommended environment:** Python 3.10 or 3.11; PyTorch with CUDA; single- or multi-GPU training.

### 2.1 Create a conda environment

```bash
conda create -n ovd_yoloe python=3.10 -y
conda activate ovd_yoloe
```

### 2.2 Install PyTorch

Install PyTorch from the official index according to your CUDA version.

Example (CUDA 12.1):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 Install project dependencies

```bash
pip install -r requirements.txt
```

### 2.4 Verify the YOLOE stack

```bash
python scripts/model_info.py --weights yoloe-11l-seg.pt
```

Successful automatic weight download and printed network statistics indicate a correct installation.

---

## 3. Dataset directory layout

### 3.1 COCO

```text
/data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### 3.2 LVIS

LVIS annotations are typically paired with COCO image directories:

```text
/data/lvis/
└── annotations/
    ├── lvis_v1_train.json
    └── lvis_v1_val.json

/data/coco/
├── train2017/
└── val2017/
```

---

## 4. Format conversion

Ultralytics training expects YOLO-style text labels. Conversion utilities are provided for COCO and LVIS annotations.

### 4.1 COCO

```bash
python tools/convert_coco_to_yolo_det.py \
  --ann-json /data/coco/annotations/instances_train2017.json \
  --image-root /data/coco/train2017 \
  --output-root /data/converted/coco_ovd \
  --split train2017

python tools/convert_coco_to_yolo_det.py \
  --ann-json /data/coco/annotations/instances_val2017.json \
  --image-root /data/coco/val2017 \
  --output-root /data/converted/coco_ovd \
  --split val2017

python tools/make_dataset_yaml.py \
  --train-meta /data/converted/coco_ovd/meta_train2017.json \
  --val-meta /data/converted/coco_ovd/meta_val2017.json \
  --dataset-root /data/converted/coco_ovd \
  --output-yaml /data/converted/coco_ovd/coco_ovd.yaml
```

### 4.2 LVIS

```bash
python tools/convert_lvis_to_yolo_det.py \
  --ann-json /data/lvis/annotations/lvis_v1_train.json \
  --image-root /data/coco \
  --output-root /data/converted/lvis_ovd \
  --split lvis_train

python tools/convert_lvis_to_yolo_det.py \
  --ann-json /data/lvis/annotations/lvis_v1_val.json \
  --image-root /data/coco \
  --output-root /data/converted/lvis_ovd \
  --split lvis_val

python tools/make_dataset_yaml.py \
  --train-meta /data/converted/lvis_ovd/meta_lvis_train.json \
  --val-meta /data/converted/lvis_ovd/meta_lvis_val.json \
  --dataset-root /data/converted/lvis_ovd \
  --output-yaml /data/converted/lvis_ovd/lvis_ovd.yaml
```

---

## 5. Training

### 5.1 Procedure

The default pipeline proceeds as follows:

1. Initialize the detector from `yoloe-11l.yaml`  
2. Load pretrained weights from `yoloe-11l-seg.pt`  
3. Fine-tune with `YOLOEPETrainer` for detection  
4. Train on COCO, then continue fine-tuning on LVIS  

This schedule is intended to preserve open-vocabulary behaviour while remaining within the target parameter budget and yielding competitive COCO/LVIS AP.

### 5.2 COCO training

```bash
python scripts/train.py \
  --data /data/converted/coco_ovd/coco_ovd.yaml \
  --model-cfg yoloe-11l.yaml \
  --pretrained yoloe-11l-seg.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --project runs/ovd \
  --name coco_yoloe_11l_det \
  --amp
```

### 5.3 LVIS fine-tuning

```bash
python scripts/train.py \
  --data /data/converted/lvis_ovd/lvis_ovd.yaml \
  --model-cfg yoloe-11l.yaml \
  --pretrained runs/ovd/coco_yoloe_11l_det/weights/best.pt \
  --epochs 40 \
  --imgsz 640 \
  --batch 16 \
  --workers 8 \
  --device 0 \
  --project runs/ovd \
  --name lvis_yoloe_11l_det \
  --amp
```

**Note:** Reduce `batch` (e.g., to 4 or 8) or `imgsz` (e.g., to 512) if GPU memory is limited.

---

## 6. Official AP / AR evaluation

Evaluation uses the **COCO API** and **LVIS API** to report standard metrics (`AP`, `AP50`, `AP75`, `AR`, etc.), rather than relying solely on in-framework validation scores. This facilitates direct comparison with published results.

### 6.1 COCO

```bash
python scripts/val.py \
  --weights runs/ovd/coco_yoloe_11l_det/weights/best.pt \
  --gt-json /data/coco/annotations/instances_val2017.json \
  --image-root /data/coco/val2017 \
  --dataset coco \
  --out-json runs/ovd/coco_preds.json \
  --metrics-json runs/ovd/coco_metrics.json \
  --imgsz 640 \
  --device 0 \
  --half
```

### 6.2 LVIS

```bash
python scripts/val.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --gt-json /data/lvis/annotations/lvis_v1_val.json \
  --image-root /data/coco \
  --dataset lvis \
  --out-json runs/ovd/lvis_preds.json \
  --metrics-json runs/ovd/lvis_metrics.json \
  --imgsz 640 \
  --device 0 \
  --half
```

**Outputs:**

- `runs/ovd/coco_preds.json`, `runs/ovd/coco_metrics.json`  
- `runs/ovd/lvis_preds.json`, `runs/ovd/lvis_metrics.json`  

---

## 7. Open-vocabulary inference

Inference accepts arbitrary text prompts; it is not restricted to the training label set.

### 7.1 Single image or directory

```bash
python scripts/predict.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --source demo.jpg \
  --classes person bus helmet fire_extinguisher backpack \
  --imgsz 640 \
  --device 0 \
  --save
```

### 7.2 Class list from file

Example `classes.txt`:

```text
person
helmet
backpack
fire extinguisher
forklift
```

```bash
python scripts/predict.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --source /data/test_images \
  --class-file classes.txt \
  --imgsz 640 \
  --device 0 \
  --save
```

---

## 8. Throughput (FPS) benchmarking

Prepare `bench_images.txt` with one absolute image path per line, for example:

```text
/data/coco/val2017/000000000139.jpg
/data/coco/val2017/000000000285.jpg
/data/coco/val2017/000000000632.jpg
```

Run:

```bash
python scripts/benchmark.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --image-list bench_images.txt \
  --classes person car dog backpack helmet \
  --imgsz 640 \
  --device 0 \
  --half \
  --warmup 20 \
  --repeats 200 \
  --out-json runs/ovd/benchmark.json
```

Example output:

```json
{
  "num_images": 100.0,
  "warmup": 20.0,
  "repeats": 200.0,
  "mean_latency_s": 0.0141,
  "fps": 70.9
}
```

---

## 9. ONNX export

For deployment, fix a set of open-vocabulary classes, then export to ONNX.

```bash
python scripts/export.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --classes person bicycle car bus truck traffic_light stop_sign \
  --format onnx \
  --imgsz 640 \
  --half \
  --dynamic
```

---

## 10. Suggested experimental configurations

### Configuration A (balanced baseline)

- Architecture: YOLOE-11L  
- Parameters: on the order of 26M  
- Input size: 640  
- Training: COCO → LVIS  
- Mixed precision: enabled  
- Evaluation: official COCO/LVIS APIs  
- Throughput: measure on a single GPU (e.g., T4, A10, RTX 4090) in PyTorch or ONNX  

### Configuration B (higher LVIS AP)

Replace defaults with `yoloe-26l.yaml` and `yoloe-26l-seg.pt`; update `--model-cfg` and `--pretrained` in the training commands accordingly.

---

## 11. Metric correspondence

| Quantity | Command / script |
|----------|------------------|
| Parameter count | `python scripts/model_info.py --weights yoloe-11l-seg.pt` |
| COCO AP / AR | `python scripts/val.py --dataset coco ...` |
| LVIS AP / AR | `python scripts/val.py --dataset lvis ...` |
| FPS | `python scripts/benchmark.py ...` |

---

## 12. Design rationale (FAQ)

**Why build on an existing detector rather than a custom backbone and head from scratch?**  
The goal is a complete pipeline for open-vocabulary detection with reproducible COCO/LVIS metrics, not a minimal proof-of-concept on a toy dataset.

**Why convert annotations to YOLO text format for training?**  
This format integrates cleanly with the Ultralytics training stack and simplifies maintenance.

**Why evaluate with official COCO/LVIS JSON and APIs?**  
Thesis-grade reporting requires standard AP/AR definitions; framework-internal metrics may not align with the literature.

**Is training optional?**  
Yes. Pretrained weights can be used for inference only, for example:

```bash
python scripts/predict.py --weights yoloe-11l-seg.pt --source demo.jpg --classes person bus helmet
```

**Could YOLO-World or other models be substituted?**  
Yes. This FYP codebase defaults to YOLOE as a practical compromise among model size, open-vocabulary capability, latency, and benchmark targets.

---

## 13. Recommended baseline for reported targets

For objectives in the neighbourhood of **~50 AP on COCO**, **~35 AP on LVIS**, **~10–70 FPS**, and **20M–80M parameters**, the following recipe is a sensible starting point:

```text
YOLOE-11L
imgsz = 640
COCO fine-tuning → LVIS fine-tuning
official COCO/LVIS evaluation
half-precision throughput benchmark
```

This path typically yields more reproducible experiments than implementing a bespoke CLIP-FCOS or DETR-style system from scratch within a single FYP cycle.

---

## Citation and acknowledgment

This repository is **Final Year Project (FYP) coursework** and builds on public models and datasets. We cite the **YOLOE** method, the **Ultralytics** implementation used in this project, and the **COCO** / **LVIS** benchmarks in the FYP report. Software versions and dataset access or download dates are recorded from our experimental setup (`pip show ultralytics`, repository commit hash, or download date).

**Licenses.** Ultralytics is distributed under **AGPL-3.0**; COCO and LVIS have their own terms on the official dataset pages. This codebase does not supersede those obligations; dataset acknowledgements follow departmental or programme requirements as applicable.
