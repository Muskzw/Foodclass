# Quick Start Guide

Get your food classification system up and running in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Dataset

### Option A: Use Sample Structure

```bash
python scripts/download_sample_data.py
```

Then add your images to the created folders:
- `data/train/class_name/`
- `data/val/class_name/`
- `data/test/class_name/`

### Option B: Download Food-101 from Kaggle

```bash
# Install kaggle CLI first: pip install kaggle
kaggle datasets download -d datatang/datatang-food101
unzip datatang-food101.zip
# Organize into train/val/test structure
```

## Step 3: Train Model

```bash
python src/train.py --data_dir data --model resnet18 --epochs 10
```

This will:
- Train a ResNet18 model
- Save the best model to `models/best_model.pth`
- Generate training curves and confusion matrix

## Step 4: Start API Server

```bash
python scripts/run_api.py
```

Or:
```bash
uvicorn api.main:app --reload
```

API will be available at `http://localhost:8000`

## Step 5: Open Web Interface

1. Open `frontend/index.html` in your browser
2. Or serve it:
   ```bash
   python -m http.server 8080
   ```
3. Navigate to `http://localhost:8080`
4. Upload a food image and see predictions!

## Testing the API

### Using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg"
```

### Using Python:

```python
import requests

with open('pizza.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', files={'file': f})
    print(response.json())
```

## Common Issues

**Model not found error:**
- Make sure you've trained a model first
- Check that `models/best_model.pth` exists

**No data found:**
- Verify your dataset structure matches the expected format
- Check that images are in the correct folders

**API connection error:**
- Ensure the API server is running
- Check the port (default: 8000)

## Next Steps

- Experiment with different models (ResNet50, CNN)
- Adjust hyperparameters (learning rate, batch size)
- Add more food classes to your dataset
- Deploy to cloud for production use

For detailed documentation, see [README.md](README.md)

