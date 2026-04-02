from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import base64
import numpy as np
import cv2

app = FastAPI()

# ✅ Load your trained model
model = YOLO("model.pt")   # your uploaded file

# 👇 This matches Lovable input
class ImageInput(BaseModel):
    image: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: ImageInput):

    try:
        # 🔹 decode base64 image
        img_bytes = base64.b64decode(data.image)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(img)

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                detections.append({
                    "label": model.names[cls],   # 🔥 IMPORTANT
                    "confidence": conf,
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1
                })

        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}
