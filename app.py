from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import io
import torch
import os

from inference_damage import load_checkpoint, predict_image_bytes
from inference_dirty import predict_image_path

MODEL_ID = "beingamit99/car_damage_detection"
TAU = 0.35  

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device).eval()
id2label = model.config.id2label

app = FastAPI(title="Car Damage (6-class) → Damaged/Intact", version="0.1")


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/damage")
async def damage(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    idx = int(probs.argmax())
    pred_label = id2label[idx]
    pred_score = float(probs[idx])

    damaged = bool(pred_score >= TAU)

    topk = np.argsort(-probs)[:3].tolist()
    top = [{"label": id2label[i], "score": float(probs[i])} for i in topk]

    return {
        "damaged": damaged,
        "threshold": TAU,
        "pred_label": pred_label,
        "pred_score": pred_score,
        "top3": top,
    }


@app.post("/damage_local")
async def damage_local(image: UploadFile = File(...)):
    ckpt_path = os.path.join("models", "damage_binary.pt")
    if not os.path.exists(ckpt_path):
        return {"error": "Local checkpoint not found. Train with train_damage.py first.", "expected": ckpt_path}
    model_local, tf, class_to_idx, damage_index = load_checkpoint(ckpt_path)
    image_bytes = await image.read()
    result = predict_image_bytes(model_local, tf, image_bytes, damage_index)
    return result


@app.post("/dirty_local")
async def dirty_local(image: UploadFile = File(...)):
    ckpt_path = os.path.join("models", "dirty_binary.pt")
    if not os.path.exists(ckpt_path):
        return {"error": "Local checkpoint not found. Train with train_dirty.py first.", "expected": ckpt_path}
    # Save uploaded image to a temporary file for convenience
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    tmp_path = os.path.join("/tmp", f"dirty_{os.getpid()}.jpg")
    with open(tmp_path, "wb") as f:
        f.write(buf.getvalue())
    try:
        result = predict_image_path(ckpt_path, tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return result

from fastapi import FastAPI, UploadFile, File
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import io
import torch
import os

from inference_damage import load_checkpoint, predict_image_bytes
from inference_dirty import predict_image_path

MODEL_ID = "beingamit99/car_damage_detection"
TAU = 0.35  

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(device).eval()
id2label = model.config.id2label

app = FastAPI(title="Car Damage (6-class) → Damaged/Intact", version="0.1")


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/damage")
async def damage(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    idx = int(probs.argmax())
    pred_label = id2label[idx]
    pred_score = float(probs[idx])

    damaged = bool(pred_score >= TAU)

    topk = np.argsort(-probs)[:3].tolist()
    top = [{"label": id2label[i], "score": float(probs[i])} for i in topk]

    return {
        "damaged": damaged,
        "threshold": TAU,
        "pred_label": pred_label,
        "pred_score": pred_score,
        "top3": top,
    }


@app.post("/damage_local")
async def damage_local(image: UploadFile = File(...)):
    ckpt_path = os.path.join("models", "damage_binary.pt")
    if not os.path.exists(ckpt_path):
        return {"error": "Local checkpoint not found. Train with train_damage.py first.", "expected": ckpt_path}
    model_local, tf, class_to_idx, damage_index = load_checkpoint(ckpt_path)
    image_bytes = await image.read()
    result = predict_image_bytes(model_local, tf, image_bytes, damage_index)
    return result


@app.post("/dirty_local")
async def dirty_local(image: UploadFile = File(...)):
    ckpt_path = os.path.join("models", "dirty_binary.pt")
    if not os.path.exists(ckpt_path):
        return {"error": "Local checkpoint not found. Train with train_dirty.py first.", "expected": ckpt_path}
    # Save uploaded image to temp buffer for predict_image_path
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    # Write to a transient tmp path
    tmp_path = os.path.join("/tmp", f"dirty_{os.getpid()}.jpg")
    with open(tmp_path, "wb") as f:
        f.write(buf.getvalue())
    try:
        result = predict_image_path(ckpt_path, tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return result

