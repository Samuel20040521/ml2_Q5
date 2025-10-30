# 🧠 Machine Learning Assignment 2 - Question 5  
**Student ID:** b11202015  
**Author:** 鄧恩陞  

---

## 📦 Environment Overview

This project was executed on **RunPod** using the following pod configuration:

| Resource | Specification |
|-----------|---------------|
| **GPU** | NVIDIA RTX 3090 × 1 |
| **vCPU** | 32 cores |
| **Memory** | 125 GB |
| **Container Disk** | 20 GB |
| **Volume Storage** | 50 GB (mounted at `/workspace`) |
| **Container Image** | `docker.io/kds285/minizero:latest` |
| **Template** | `j1g0pows5q` |
| **Uptime** | ~11 hours |
| **Total Cost** | ≈ $0.4697/hr |
| **Start Command** | `/bin/bash -lc "sleep infinity"` |

---

## ⚙️ Setup Workflow (手寫重灌流程)

The following steps describe how to set up the environment and reproduce the training pipeline from scratch.

### 1. Clone the Teaching Assistant’s Repository
```bash
git clone https://github.com/gcobs104628/ML-Assignment2-Q5.git
````

### 2. Replace the Core Code with Personal Repository

```bash
cd /workspace/ML-Assignment2-Q5/style_detection

rm -rf code

git clone https://github.com/Samuel20040521/ml2_Q5.git code

mv code/requirements.txt /workspace/ML-Assignment2-Q5/

cd ..

pip install -r requirements.txt
```

---

### 3. Download the Dataset from Kaggle

```bash
mkdir -p ~/.kaggle

echo '{"username":"enshengteng","key":"XXXXXXXX"}' > ~/.kaggle/kaggle.json

chmod 600 ~/.kaggle/kaggle.json

pip install kaggle

cd /workspace/ML-Assignment2-Q5

kaggle competitions download -c machine-learning-class-fall-2025-assignment-2-q-5

unzip machine-learning-class-fall-2025-assignment-2-q-5.zip
```

---

### 4. Build the Project

```bash
./scripts/build.sh go
```

---

### 5. Verify GPU Availability

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch =", torch.__version__, " / torch.version.cuda =", torch.version.cuda)
print("torch.cuda.is_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU =", torch.cuda.get_device_name(0))
PY
```

---

## 🧩 Training Pipeline

### 1. Environment Variables

Before training, set up environment paths:

```bash
export PYTHONPATH="$PWD:$PWD/build:$PWD/style_detection/code"
export PYTHONDONTWRITEBYTECODE=1
```

---

### 2. Start Training

Run the following command to train the model using the **ultra preset**:

```bash
python -m style_detection.code.train_launcher \
>   --preset ultra \
>   --device cuda \
>   --run_name style4090 \
>   --save_dir checkpoints \
>   --batch_size 48 \
>   --eval_batch_size 96 \
>   --epochs 30 \
>   --num_workers 8 \
>   --keep_last 3 \
>   --ge2e_weight 0.1
```

> 📝 The trained weight file will be saved to `checkpoints/style4090_epoch30.pt`.

---

## 🔍 Inference (Generating Submission)

Use the trained model to generate your `submission.csv`:

```bash
python style_detection/code/Q5.py \
>   --mode infer \
>   --conf conf.cfg \
>   --model_path checkpoints/style4090_epoch30.pt \
>   --submission_path submission.csv
```

---

## 🚀 Kaggle Submission

Once `submission.csv` is generated, submit it to the competition leaderboard:

```bash
kaggle competitions submit -c machine-learning-class-fall-2025-assignment-2-q-5 -f submission.csv -m "3090-1"
```

---

## 📁 Directory Overview

```
b11202015/
├── Q5.py                         # Main inference script
├── train_launcher.py             # Training code
├── style4090_epoch30.pt          # Pretrained weight file
├── requirements.txt              # Python dependencies
└── readme.md                     # This document
```
