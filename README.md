# OVD-YOLOE-Light

一个可直接落地的 **open-vocabulary object detection** 轻量工程，默认基于 **YOLOE-11L** 做开放词汇目标检测。

这个工程的目标是：
- 模型参数量落在 **20M–80M** 区间内
- 支持 **COCO / LVIS** 数据集
- 支持 **开放词汇推理**（任意文本类别 prompt）
- 支持 **训练 / 官方 AP-AR 评估 / FPS 测速 / ONNX 导出**

> 默认推荐模型：`yoloe-11l-seg.pt` + `yoloe-11l.yaml`
>
> 这样做的原因是它在轻量级范围内更容易同时兼顾速度、精度和开放词汇能力。

---

## 1. 工程结构

```text
ovd_yoloe_light/
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

## 2. 环境安装

建议环境：
- Python 3.10 / 3.11
- PyTorch + CUDA
- 单卡或多卡均可

### 2.1 新建环境

```bash
conda create -n ovd_yoloe python=3.10 -y
conda activate ovd_yoloe
```

### 2.2 安装 PyTorch

请按你的 CUDA 版本安装官方 PyTorch。

例如 CUDA 12.1：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 安装依赖

```bash
pip install -r requirements.txt
```

### 2.4 验证 YOLOE 是否可用

```bash
python scripts/model_info.py --weights yoloe-11l-seg.pt
```

如果这里能自动下载权重并打印网络信息，说明环境 OK。

---

## 3. 数据集目录约定

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

LVIS 通常复用 COCO 图像目录：

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

## 4. 数据转换

Ultralytics 训练时更适合使用 YOLO txt 标签格式，因此这里提供了 COCO / LVIS 的转换脚本。

### 4.1 转 COCO

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

### 4.2 转 LVIS

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

## 5. 训练

### 5.1 训练思路

这里采用：
1. 用 `yoloe-11l.yaml` 初始化检测模型
2. 加载 `yoloe-11l-seg.pt` 作为预训练权重
3. 用 `YOLOEPETrainer` 做 detection 微调
4. 先跑 COCO，再在 LVIS 上继续微调

这样更符合你的需求：
- 参数量控制在轻量范围内
- 保留开放词汇能力
- 更容易达到 COCO / LVIS 的 AP 目标区间

### 5.2 训练 COCO

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

### 5.3 再训练 LVIS

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

> 如果你的显存不够，把 `batch` 改成 4 或 8；必要时把 `imgsz` 改成 512。

---

## 6. 官方 AP / AR 评估

这个工程没有只依赖框架内部的 val 指标，而是额外写了 **官方 COCO API / LVIS API** 的评估脚本，直接输出 `AP / AP50 / AP75 / AR...`，更适合你写实验。

### 6.1 评估 COCO

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

### 6.2 评估 LVIS

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

输出会写到：
- `runs/ovd/coco_preds.json`
- `runs/ovd/coco_metrics.json`
- `runs/ovd/lvis_preds.json`
- `runs/ovd/lvis_metrics.json`

---

## 7. 开放词汇推理

你可以不受训练类别限制，直接输入文本类别 prompt。

### 7.1 单图 / 文件夹推理

```bash
python scripts/predict.py \
  --weights runs/ovd/lvis_yoloe_11l_det/weights/best.pt \
  --source demo.jpg \
  --classes person bus helmet fire_extinguisher backpack \
  --imgsz 640 \
  --device 0 \
  --save
```

### 7.2 用类别文件推理

`classes.txt`:

```text
person
helmet
backpack
fire extinguisher
forklift
```

然后：

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

## 8. FPS 测速

先准备一个 `bench_images.txt`，每行一张图片绝对路径，例如：

```text
/data/coco/val2017/000000000139.jpg
/data/coco/val2017/000000000285.jpg
/data/coco/val2017/000000000632.jpg
```

运行：

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

输出类似：

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

## 9. 导出 ONNX

如果你要部署，可以先固定一组开放词汇类别，再导出 ONNX。

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

## 10. 推荐实验配置

### 方案 A：更稳妥地贴近你的指标

- Backbone / model family: `YOLOE-11L`
- Params: 约 26M 级别
- Input size: `640`
- 训练顺序：`COCO -> LVIS`
- Mixed precision: 开
- Eval: 官方 COCO / LVIS API
- Benchmark: 单卡 T4 / A10 / 4090 测 PyTorch 或 ONNX

### 方案 B：更追求 LVIS AP

把默认模型替换为：
- `yoloe-26l.yaml`
- `yoloe-26l-seg.pt`

你只需要把训练命令里的 `--model-cfg` 和 `--pretrained` 改掉即可。

---

## 11. 你最关心的几个结果怎么对应

### 参数量

用：

```bash
python scripts/model_info.py --weights yoloe-11l-seg.pt
```

### COCO AP / AR

用：

```bash
python scripts/val.py --dataset coco ...
```

### LVIS AP / AR

用：

```bash
python scripts/val.py --dataset lvis ...
```

### FPS

用：

```bash
python scripts/benchmark.py ...
```

---

## 12. 常见问题

### Q1. 为什么不用纯从零写 backbone + head？
因为你要的是“完整能跑、能做 open-vocabulary detection、还能冲 COCO/LVIS 指标”的工程，不是一个只能在 toy dataset 上跑通的 demo。

### Q2. 为什么训练时还是要转换成 YOLO 标签？
因为这样最稳定，最方便接 Ultralytics 训练管线，也更容易自己维护。

### Q3. 为什么评估时又回到 COCO/LVIS 官方 json？
因为你最终论文里需要的是官方 AP / AR，不是框架内部的近似值。

### Q4. 能不能只做推理不训练？
可以。
你直接运行：

```bash
python scripts/predict.py --weights yoloe-11l-seg.pt --source demo.jpg --classes person bus helmet
```

### Q5. 能不能换成 YOLO-World？
可以，但这份工程默认选 YOLOE，是因为在“轻量 + 开放词汇 + 实时 + 指标目标”这几个条件下更贴近你的要求。

---

## 13. 最后建议

如果你想尽量贴近：
- **COCO 50 AP**
- **LVIS 35 AP**
- **10–70 FPS**
- **20M–80M 参数量**

优先跑这套：

```text
YOLOE-11L
imgsz=640
COCO fine-tune -> LVIS fine-tune
official COCO/LVIS eval
half precision benchmark
```

这会比你从零手写一个 CLIP + FCOS / DETR 混合模型更容易真正落到可复现实验。
