"""
frontend/app.py — Gradio UI that calls the FastAPI backend.
Run: python frontend/app.py
"""
import requests
import gradio as gr
from PIL import Image

API_URL = "http://localhost:8000/predict"

CLASS_LABELS = {
    "Glioma":     ("🔴", "High-grade malignant brain tumor originating in glial cells."),
    "Meningioma": ("🟡", "Usually benign tumor arising from the meninges."),
    "No Tumor":   ("🟢", "No tumor detected in the MRI scan."),
    "Pituitary":  ("🔵", "Tumor located in the pituitary gland, often benign."),
}

def classify(image: Image.Image):
    if image is None:
        return "Upload an MRI scan first.", {}, "", ""

    # Send to FastAPI
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        resp = requests.post(API_URL, files={"file": ("scan.png", buf, "image/png")}, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return f"❌ API Error: {e}", {}, "", ""

    data      = resp.json()
    label     = data["label"]
    icon, desc = CLASS_LABELS[label]
    confidence = data["confidence"] * 100
    latency   = data["latency_ms"]
    scores    = {k: v for k, v in data["scores"].items()}

    diagnosis = f"{icon} **{label}** — {confidence:.1f}% confidence"
    info      = f"_{desc}_\n\n⚡ Prediction time: **{latency} ms**"
    return diagnosis, scores, info

# ── UI ───────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="🧠 MRI Brain Tumor Classifier",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="purple"),
    css="""
    .gradio-container { max-width: 900px; margin: auto; }
    footer { display: none !important; }
    """,
) as demo:

    gr.Markdown("""
    # 🧠 MRI Brain Tumor Classifier
    **EfficientNetB2 · 97%+ Accuracy · Grad-CAM Ready · FastAPI Backend**
    
    Upload an MRI brain scan to classify: Glioma · Meningioma · Pituitary · No Tumor
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="📤 Upload MRI Scan", height=300)
            btn = gr.Button("🔍 Classify", variant="primary", size="lg")

        with gr.Column(scale=1):
            diagnosis  = gr.Markdown(label="Diagnosis")
            bar_chart  = gr.Label(label="📊 Class Probabilities", num_top_classes=4)
            info_box   = gr.Markdown(label="Clinical Info")

    btn.click(fn=classify, inputs=img_input, outputs=[diagnosis, bar_chart, info_box])

    gr.Examples(
        examples=[["examples/glioma_sample.jpg"],
                  ["examples/notumor_sample.jpg"]],
        inputs=img_input,
        label="Try Sample Scans",
    )

    gr.Markdown("---\nBuilt by **Muhammad Zeeshan Malik** · [GitHub](https://github.com/AImindcrafter) · [Portfolio](https://aimindcrafter.github.io)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
