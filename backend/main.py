"""
backend/main.py — FastAPI inference server for MRI Brain Tumor Classifier.
Run: uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
import io
import time
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "EfficientNetB2_brain_tumor_model.keras"   # existing trained model
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE    = (224, 224)

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="🧠 MRI Tumor Classifier API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Model (loaded once at startup) ──────────────────────────────────────────
model = None
predict_fn = None   # tf.function compiled for speed

@app.on_event("startup")
def load_model():
    global model, predict_fn
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Compile to graph — ~30% faster than eager mode
    @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32)])
    def _infer(x):
        return model(x, training=False)

    predict_fn = _infer

    # Warm-up: eliminates first-request latency
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    for _ in range(3):
        predict_fn(dummy)

    print("✅ Model ready.")

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file.")

    # Preprocess
    img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)

    # Infer
    t0 = time.perf_counter()
    probs = predict_fn(arr).numpy()[0]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    idx = int(np.argmax(probs))
    return {
        "label":      CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]), 4),
        "scores":     {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(4)},
        "latency_ms": latency_ms,
    }
