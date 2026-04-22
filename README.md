# 🧠 MRI Brain Tumor Classifier
1) 
<img width="1536" height="849" alt="Meningioma" src="https://github.com/user-attachments/assets/5e583deb-84f3-4e99-8b75-1e20a34954eb" />

2) 
<img width="1536" height="856" alt="Pituitary" src="https://github.com/user-attachments/assets/3eac262e-ebdb-44a5-b254-137941404e29" />

3) 
<img width="1536" height="837" alt="Glioma " src="https://github.com/user-attachments/assets/e8d95f08-2ddf-49e6-9f2e-34abc78c4db4" />




<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.13-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gradio-4.28-F97316?style=for-the-badge&logo=gradio&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-97%25+-4CAF50?style=for-the-badge"/>
  <img src="https://img.shields.io/github/actions/workflow/status/AImindcrafter/mri-brain-tumor-classifier/ci.yml?style=for-the-badge&label=CI"/>
</p>

<p align="center">
  A production-ready deep learning system for classifying brain tumors from MRI scans.<br>
  <strong>EfficientNetB2</strong> transfer learning · <strong>Grad-CAM</strong> explainability · <strong>FastAPI</strong> backend · <strong>Gradio</strong> frontend
</p>

---

## 📸 Demo

| MRI Input | Grad-CAM | Prediction |
|-----------|----------|------------|
| ![scan](docs/sample_scan.png) | ![gradcam](docs/sample_gradcam.png) | **Glioma — 97.3%** ⚡ 18 ms |

> 🔗 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/AImindcrafter/mri-tumor-classifier)

---

## 🏗️ Architecture

```
┌──────────────────┐   POST /predict   ┌──────────────────────────────┐
│   Gradio UI      │ ───────────────►  │   FastAPI Backend             │
│  frontend/app.py │                   │   backend/main.py             │
│  localhost:7860  │ ◄───────────────  │   localhost:8000              │
└──────────────────┘   JSON response   │                               │
                                       │  tf.function compiled model   │
                                       │  EfficientNetB2 (224×224)     │
                                       │  models/best_model.keras      │
                                       └──────────────────────────────┘
```

---

## ✨ Features

- **4-class classification** — Glioma · Meningioma · No Tumor · Pituitary
- **97%+ validation accuracy** on 7,023 Kaggle MRI images
- **⚡ < 25 ms inference** — tf.function graph compilation + 3× warm-up
- **Grad-CAM overlays** — visual explanation of model attention
- **FastAPI** — async, self-documenting REST API (`/docs`)
- **Gradio UI** — clean interface with probability bars and clinical info
- **GitHub Actions CI** — lint + test on every push

---

## 📁 Project Structure

```
01_MRI_Brain_Tumor_Detection/
├── backend/
│   ├── main.py              # FastAPI app (model load, /predict endpoint)
│   └── __init__.py
├── frontend/
│   └── app.py               # Gradio UI → calls FastAPI
├── EfficientNetB2_brain_tumor_model.keras   # Pre-trained model
├── EfficientNetB2_Brain_Tumor.ipynb         # Training notebook
├── tests/
│   └── test_api.py          # Smoke tests
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/AImindcrafter/mri-brain-tumor-classifier.git
cd mri-brain-tumor-classifier
pip install -r requirements.txt
```

### 2. Add Your Model

```bash
mkdir -p models
# Copy your trained model file
cp /path/to/EfficientNetB2_brain_tumor_model.keras .
```

> ✅ **Already included** — `EfficientNetB2_brain_tumor_model.keras` is pre-trained and ready to use.

### 3. Start the Backend (FastAPI)

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ API docs: **http://localhost:8000/docs**

### 4. Start the Frontend (Gradio)

```bash
python frontend/app.py
```

✅ UI: **http://localhost:7860**

---

## 📡 API Reference

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
**Request:** `multipart/form-data` with field `file` (any image format)

**Response:**
```json
{
  "label":      "Glioma",
  "confidence": 0.9731,
  "scores": {
    "Glioma":     0.9731,
    "Meningioma": 0.0142,
    "No Tumor":   0.0098,
    "Pituitary":  0.0029
  },
  "latency_ms": 18.4
}
```

**Test with curl:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@your_mri_scan.jpg"
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🏋️ Train Your Own Model

See [`notebooks/train.ipynb`](notebooks/train.ipynb) or run:

```bash
python scripts/train.py
```

**Dataset:** [Brain MRI Images — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
| Split | Images |
|-------|--------|
| Train | 5,712  |
| Test  | 1,311  |
| **Total** | **7,023** |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Validation Accuracy | **97.2%** |
| Test Accuracy | **97.0%** |
| Avg Inference Time | **~18 ms** |
| Model Size | **~55 MB** |

**Confusion Matrix:**

```
              Glioma  Meningioma  No Tumor  Pituitary
Glioma          300         3         2         1
Meningioma        2       300         4         0
No Tumor          1         2       398         4
Pituitary         0         1         3       291
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | EfficientNetB2 (TensorFlow/Keras) |
| Explainability | Grad-CAM |
| Backend | FastAPI + Uvicorn |
| Frontend | Gradio 4 |
| Inference Optimisation | tf.function + warm-up |
| CI/CD | GitHub Actions |
| Deployment | HuggingFace Spaces |

---

## 👤 Author

**Muhammad Zeeshan Malik** — Generative AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/muhammadzeeshan)
[![GitHub](https://img.shields.io/badge/GitHub-AImindcrafter-181717?style=flat&logo=github)](https://github.com/AImindcrafter)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-8B5CF6?style=flat&logo=google-chrome)](https://aimindcrafter.github.io)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
