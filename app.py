from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformer import TumorClassifierViT   # your ViT model class

# ---------------------
# CONFIG
# ---------------------
MODEL_PATH = r"G:\braintumer\best_model.pth"   # Windows path to your model
class_names = ["meningioma", "glioma", "notumor", "pituitary"]

# Preprocessing for inference (no random augmentations)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ---------------------
# Load model
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TumorClassifierViT(num_classes=len(class_names))
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---------------------
# FastAPI app
# ---------------------
app = FastAPI(title="Brain Tumor Detection API (ViT)")

@app.get("/")
def root():
    return {"message": "API is running", "classes": class_names}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        img_t = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)

        return JSONResponse({
            "prediction": class_names[pred_class.item()],
            "confidence": float(confidence.item())
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
