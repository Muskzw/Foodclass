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
from typing import List, Dict
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
    # Always include original
    images.append(image)
    # Basic TTA: original + horizontal flip
    images.append(F.hflip(image))
    if TTA_MODE == 'extended':
        # Add 90/180/270 degree rotations
        images.append(F.rotate(image, 90))
        images.append(F.rotate(image, 180))
        images.append(F.rotate(image, 270))
    # Transform and stack into batch
    tensors = [transform(img) for img in images]
    batch = torch.stack(tensors, dim=0)
    return batch


def predict_with_tta(image_bytes: bytes) -> Dict:
    """Make prediction on preprocessed image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train a model first.")
    
    model.eval()
    with torch.no_grad():
        # Prepare TTA batch
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        batch = make_tta_tensors(image).to(device)
        # Forward pass for all views, average logits
        logits = model(batch)  # shape: [N, C]
        mean_logits = logits.mean(dim=0)
        probabilities = torch.nn.functional.softmax(mean_logits, dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities, min(3, len(class_names)))
        
        predictions = []
        for prob, idx in zip(top3_probs, top3_indices):
            predictions.append({
                'class': class_names[idx.item()],
                'confidence': prob.item()
            })
        
        result = {
            'predicted_class': class_names[predicted_idx.item()],
            'confidence': confidence.item(),
            'top_predictions': predictions
        }

        # Reject low-confidence predictions as out-of-distribution
        if confidence.item() < CONF_THRESHOLD:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Image appears to be outside the trained dataset classes",
                    "max_confidence": round(confidence.item(), 4),
                    "threshold": CONF_THRESHOLD,
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
async def predict_image(file: UploadFile = File(...)):
    """
    Predict food class from uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        JSON with predicted class, confidence, and top predictions
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Predict with multi-angle TTA
        result = predict_with_tta(image_bytes)
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
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
            result = predict_with_tta(image_bytes)
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

