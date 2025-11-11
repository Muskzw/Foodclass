"""
FastAPI application for food classification API.
Provides RESTful endpoints for image classification.
"""

import sys
import os

# Add parent directory to path to allow imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import io
from typing import List, Dict, Optional
import uvicorn

from src.model import load_model

app = FastAPI(
    title="Food Classification API",
    description="RESTful API for food image classification using deep learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
class_names = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Confidence threshold for accepting predictions (configurable via env)
CONF_THRESHOLD = float(os.getenv('PREDICTION_CONF_THRESHOLD', '0.6'))
# Test-time augmentation mode: 'basic' (orig + hflip) or 'extended' (+ 90/180/270)
TTA_MODE = os.getenv('PREDICTION_TTA', 'extended').lower()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])


def load_model_on_startup():
    """Load the trained model on application startup."""
    global model, class_names
    
    model_path = os.getenv('MODEL_PATH', 'models/best_model.pth')
    model_name = os.getenv('MODEL_NAME', 'resnet18')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please train a model first or set MODEL_PATH environment variable")
        return
    
    try:
        model, class_names = load_model(model_path, model_name, device)
        print(f"Model loaded successfully from {model_path}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model_on_startup()


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for model inference."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


def make_tta_tensors(image: Image.Image) -> torch.Tensor:
    """Create a batch tensor with TTA views based on configured mode."""
    images: List[Image.Image] = []
    # Always include original and flip
    images.append(image)
    images.append(F.hflip(image))

    if TTA_MODE in ('extended', 'max'):
        # Add 90/180/270 degree rotations
        for angle in (90, 180, 270):
            rot = F.rotate(image, angle)
            images.append(rot)

    if TTA_MODE == 'max':
        # Add small-angle rotations with flips for robustness
        small_angles = (-30, -15, 15, 30)
        for angle in small_angles:
            rot = F.rotate(image, angle)
            images.append(rot)
            images.append(F.hflip(rot))
    # Transform and stack into batch
    tensors = [transform(img) for img in images]
    batch = torch.stack(tensors, dim=0)
    return batch


def predict_with_tta(image_bytes: bytes, threshold_override: Optional[float] = None, tta_override: Optional[str] = None) -> Dict:
    """Make prediction on preprocessed image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    model.eval()
    with torch.no_grad():
        # Prepare TTA batch
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        global TTA_MODE
        mode = (tta_override or TTA_MODE).lower()
        # Temporarily switch TTA mode if override provided
        original_mode = TTA_MODE
        if tta_override is not None:
            TTA_MODE = mode
        try:
            batch = make_tta_tensors(image).to(device)
        finally:
            if tta_override is not None:
                TTA_MODE = original_mode
        # Forward pass for all views, average logits
        logits = model(batch)  # shape: [N, C]
        mean_logits = logits.mean(dim=0)
        probabilities = torch.nn.functional.softmax(mean_logits, dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
        result = {
            'predicted_class': class_names[predicted_idx.item()],
            'confidence': confidence.item()
        }

        # Reject low-confidence predictions as out-of-distribution
        thr = CONF_THRESHOLD if threshold_override is None else float(threshold_override)
        if confidence.item() < thr:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Image appears to be outside the trained dataset classes",
                    "max_confidence": round(confidence.item(), 4),
                    "threshold": thr,
                }
            )

        return result


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Food Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/predict": "Image classification (POST)",
            "/classes": "List all classes",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "num_classes": len(class_names) if class_names else 0,
        "confidence_threshold": CONF_THRESHOLD,
        "tta_mode": TTA_MODE
    }


@app.get("/classes")
async def get_classes():
    """Get list of all classification classes."""
    if not class_names:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": class_names, "num_classes": len(class_names)}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...), threshold: Optional[float] = None, tta: Optional[str] = None):
    """
    Predict food class from uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        JSON with predicted class and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Predict with multi-angle TTA and optional overrides
        result = predict_with_tta(image_bytes, threshold_override=threshold, tta_override=tta)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), threshold: Optional[float] = None, tta: Optional[str] = None):
    """
    Predict food classes for multiple images.
    
    Args:
        files: List of image files
    
    Returns:
        JSON with predictions for each image
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "Invalid file type"
            })
            continue
        
        try:
            image_bytes = await file.read()
            result = predict_with_tta(image_bytes, threshold_override=threshold, tta_override=tta)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

