# 🧠 MRI Brain Tumor Detection & Classification
**Stack:** EfficientNetB2 · TensorFlow · Grad-CAM · Gradio · HuggingFace Spaces

---

## 📋 Project Overview
A 4-class tumor classification system (Glioma, Meningioma, Pituitary, No Tumor) built with EfficientNetB2 transfer learning on 7,023 Kaggle MRI images, achieving 97%+ validation accuracy. Integrated Grad-CAM for clinical explainability and deployed as a live Gradio app.

---

## 🗺️ Project Lifecycle — A to Z

### Phase 1: Environment Setup (Day 1)
```bash
# Create virtual environment
python -m venv mri_env
source mri_env/bin/activate          # Linux/Mac

# Install dependencies
pip install tensorflow==2.13 gradio huggingface_hub kaggle \
            matplotlib seaborn scikit-learn pillow numpy opencv-python
```

**Folder Structure:**
```
01_MRI_Brain_Tumor_Detection/
├── data/
│   ├── raw/                  # Downloaded Kaggle dataset
│   └── processed/            # Resized, normalized images
├── models/
│   └── best_model.h5         # Saved best checkpoint
├── notebooks/
│   └── 01_EDA.ipynb
│   └── 02_Training.ipynb
│   └── 03_Evaluation.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── gradcam.py
│   └── app.py                # Gradio app
├── requirements.txt
└── README.md
```

---

### Phase 2: Data Acquisition & EDA (Day 1–2)

**Step 1 — Download Dataset from Kaggle:**
```bash
# Configure Kaggle API key (~/.kaggle/kaggle.json)
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data/raw/
unzip data/raw/brain-tumor-mri-dataset.zip -d data/raw/
```

**Step 2 — Exploratory Data Analysis (notebook: 01_EDA.ipynb):**
```python
import os, matplotlib.pyplot as plt
from collections import Counter

# Count per class
data_dir = "data/raw/Training"
classes = os.listdir(data_dir)
counts = {cls: len(os.listdir(f"{data_dir}/{cls}")) for cls in classes}
print(counts)  # Check class balance

# Visualize sample images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, cls in enumerate(classes):
    img_path = f"{data_dir}/{cls}/{os.listdir(f'{data_dir}/{cls}')[0]}"
    axes[i].imshow(plt.imread(img_path))
    axes[i].set_title(cls)
plt.show()
```

---

### Phase 3: Data Preprocessing (Day 2)

**src/data_loader.py:**
```python
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

def build_dataset(data_dir, augment=False):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    train_ds = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )
    val_ds = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=False
    )
    return train_ds, val_ds
```

---

### Phase 4: Model Building (Day 3)

**src/model.py — EfficientNetB2 Transfer Learning:**
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_model(num_classes=4):
    base = tf.keras.applications.EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    base.trainable = False          # Freeze base initially

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

### Phase 5: Training & Fine-Tuning (Day 3–4)

**src/train.py:**
```python
import tensorflow as tf
from model import build_model
from data_loader import build_dataset

# --- Stage 1: Train classifier head (frozen base) ---
train_ds, val_ds = build_dataset("data/raw/Training")
model = build_model(num_classes=4)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/best_model.h5",
        save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3)
]

history1 = model.fit(train_ds, validation_data=val_ds,
                     epochs=15, callbacks=callbacks)

# --- Stage 2: Fine-tune top layers ---
model.layers[1].trainable = True     # Unfreeze EfficientNetB2
for layer in model.layers[1].layers[:-30]:
    layer.trainable = False          # Keep bottom layers frozen

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model.fit(train_ds, validation_data=val_ds,
                     epochs=20, callbacks=callbacks)

print("Training Complete!")
```

---

### Phase 6: Evaluation & Grad-CAM (Day 4–5)

**src/gradcam.py:**
```python
import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed
```

---

### Phase 7: Gradio App & HuggingFace Deployment (Day 5–6)

**src/app.py:**
```python
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from gradcam import make_gradcam_heatmap, overlay_gradcam

model = tf.keras.models.load_model("models/best_model.h5")
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def predict(image):
    img = image.resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, 0)
    preds = model.predict(arr)[0]
    label = CLASS_NAMES[np.argmax(preds)]
    confidence = {CLASS_NAMES[i]: float(preds[i]) for i in range(4)}

    heatmap = make_gradcam_heatmap(arr, model)
    overlay = overlay_gradcam(np.array(img), heatmap)

    return label, confidence, Image.fromarray(overlay)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Scan"),
    outputs=[
        gr.Label(label="Diagnosis"),
        gr.Label(label="Confidence Scores"),
        gr.Image(label="Grad-CAM Visualization")
    ],
    title="🧠 MRI Brain Tumor Classifier",
    description="Upload an MRI scan to classify tumor type with Grad-CAM visualization."
)

if __name__ == "__main__":
    demo.launch()
```

**Deploy to HuggingFace Spaces:**
```bash
pip install huggingface_hub
huggingface-cli login

# Create space and push
git init && git add .
git commit -m "Initial deployment"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/mri-tumor-classifier
git push origin main
```

---

### Phase 8: Documentation & README (Day 6)

**Key metrics to document:**
- ✅ 97%+ validation accuracy
- ✅ 4-class classification
- ✅ Grad-CAM overlays for interpretability
- ✅ Live HuggingFace demo link

---

## ⏱️ Timeline Summary
| Phase | Task | Duration |
|-------|------|----------|
| 1 | Environment Setup | 2 hrs |
| 2 | Data Download + EDA | 4 hrs |
| 3 | Preprocessing | 2 hrs |
| 4 | Model Building | 3 hrs |
| 5 | Training + Fine-tuning | 4 hrs |
| 6 | Grad-CAM Evaluation | 3 hrs |
| 7 | Gradio App + Deployment | 4 hrs |
| 8 | README + Docs | 1 hr |
| **Total** | | **~23 hrs** |

---

## 🔗 Resources
- Dataset: [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- EfficientNetB2: [TF Docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications/EfficientNetB2)
- Grad-CAM Paper: [Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391)
- Gradio Docs: [gradio.app](https://www.gradio.app)
- HuggingFace Spaces: [huggingface.co/spaces](https://huggingface.co/spaces)
