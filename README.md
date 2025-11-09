# Food Classification System

A complete computer vision system for food image classification, featuring deep learning model training, FastAPI deployment, and a modern web interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Selection](#dataset-selection)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [API Deployment](#api-deployment)
- [System Integration](#system-integration)
- [Evaluation Metrics](#evaluation-metrics)
- [Ethical Considerations](#ethical-considerations)
- [Challenges and Solutions](#challenges-and-solutions)

## 🎯 Overview

This project implements an end-to-end computer vision system for classifying food images using deep learning. The system includes:

- **Model Training**: Support for CNN and ResNet architectures with transfer learning
- **RESTful API**: FastAPI backend for real-time predictions
- **Web Interface**: Modern, responsive frontend for image upload and visualization
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix, and training curves

## ✨ Features

- Multiple model architectures (Simple CNN, ResNet18, ResNet50)
- Transfer learning with ImageNet pretrained weights
- Data augmentation for improved generalization
- Real-time image classification via REST API
- Interactive web interface with drag-and-drop support
- Comprehensive evaluation metrics and visualizations
- Batch prediction support
- Model checkpointing and early stopping

## 📊 Dataset Selection

### Recommended Datasets

1. **Food-101 Dataset** (Recommended)
   - **Source**: Kaggle / ETH Zurich
   - **Size**: 101 food categories, ~101,000 images
   - **Justification**: 
     - Large, diverse dataset with high-quality images
     - Well-balanced classes
     - Standard benchmark for food classification
     - Download: `kaggle datasets download -d datatang/datatang-food101`

2. **Food-11 Dataset**
   - **Source**: Kaggle
   - **Size**: 11 food categories, ~16,000 images
   - **Justification**: 
     - Smaller dataset, faster training
     - Good for prototyping and testing
     - Well-structured for train/val/test splits

3. **Custom Dataset**
   - Create your own dataset by organizing images into class folders
   - Structure: `data/train/class_name/`, `data/val/class_name/`, `data/test/class_name/`

### Dataset Structure

```
data/
├── train/
│   ├── pizza/
│   ├── burger/
│   ├── sushi/
│   └── ...
├── val/
│   ├── pizza/
│   ├── burger/
│   └── ...
└── test/
    ├── pizza/
    ├── burger/
    └── ...
```

## 📁 Project Structure

```
Foodclass/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py             # Model architectures
│   └── train.py             # Training script
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application
├── frontend/
│   └── index.html           # Web interface
├── models/                  # Saved models (created after training)
├── data/                    # Dataset (user-provided)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone or navigate to the project directory:**
   ```bash
   cd Foodclass
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and prepare dataset:**
   - Download Food-101 or Food-11 from Kaggle
   - Organize into `data/train/`, `data/val/`, `data/test/` structure
   - Or use the provided script to download a sample dataset

## 💻 Usage

### 1. Model Training

Train a model with default settings:

```bash
python src/train.py --data_dir data --model resnet18 --epochs 20
```

**Arguments:**
- `--data_dir`: Directory containing train/val/test folders (default: `data`)
- `--model`: Model architecture - `resnet18`, `resnet50`, or `cnn` (default: `resnet18`)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save models (default: `models`)

**Example:**
```bash
python src/train.py --data_dir data --model resnet50 --batch_size 16 --epochs 30 --lr 0.0001
```

### 2. Start API Server

```bash
python api/main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 3. Use Web Interface

1. Open `frontend/index.html` in a web browser
2. Or serve it using a simple HTTP server:
   ```bash
   # Python
   python -m http.server 8080
   
   # Node.js
   npx http-server -p 8080
   ```
3. Navigate to `http://localhost:8080`
4. Upload an image to get predictions

### 4. API Endpoints

#### Health Check
```bash
GET /health
```

#### Get Classes
```bash
GET /classes
```

#### Predict Single Image
```bash
POST /predict
Content-Type: multipart/form-data
Body: file (image file)
```

**Response:**
```json
{
  "predicted_class": "pizza",
  "confidence": 0.95,
  "top_predictions": [
    {"class": "pizza", "confidence": 0.95},
    {"class": "burger", "confidence": 0.03},
    {"class": "sushi", "confidence": 0.02}
  ]
}
```

#### Batch Prediction
```bash
POST /predict/batch
Content-Type: multipart/form-data
Body: files (multiple image files, max 10)
```

## 🔬 Model Development

### Architecture Selection

1. **Simple CNN**: Custom convolutional neural network
   - Pros: Lightweight, fast training, good for small datasets
   - Cons: Lower accuracy compared to ResNet

2. **ResNet18**: Residual network with 18 layers
   - Pros: Good balance of accuracy and speed, transfer learning support
   - Cons: Moderate model size

3. **ResNet50**: Deeper residual network
   - Pros: Higher accuracy, better feature extraction
   - Cons: Larger model, slower inference

### Training Process

1. **Data Preprocessing:**
   - Resize to 224x224 pixels
   - Normalize with ImageNet statistics
   - Data augmentation (random flips, rotations, color jitter)

2. **Training Features:**
   - Transfer learning with ImageNet pretrained weights
   - Adam optimizer with learning rate scheduling
   - Early stopping to prevent overfitting
   - Model checkpointing

3. **Evaluation:**
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix visualization
   - Training/validation loss and accuracy curves

## 📈 Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and weighted average
- **Recall**: Per-class and weighted average
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance
- **Loss Curves**: Training and validation loss over epochs
- **Accuracy Curves**: Training and validation accuracy over epochs

All metrics are calculated on the test set and saved as visualizations.

## ⚠️ Challenges and Solutions

### 1. Overfitting

**Problem**: Model performs well on training data but poorly on validation/test data.

**Solutions Implemented:**
- Data augmentation (random flips, rotations, color jitter)
- Dropout layers (0.5 dropout rate)
- Weight decay (L2 regularization)
- Early stopping based on validation accuracy
- Transfer learning with pretrained weights

### 2. Class Imbalance

**Problem**: Some food classes have significantly more samples than others.

**Solutions:**
- Weighted loss function (can be added)
- Data augmentation for minority classes
- Balanced sampling strategies
- F1-score as primary metric (instead of accuracy)

### 3. Limited Training Data

**Solution:**
- Transfer learning with ImageNet pretrained models
- Data augmentation to artificially increase dataset size
- Fine-tuning only the classifier head initially

### 4. Computational Resources

**Solutions:**
- Support for CPU and GPU training
- Batch size adjustment
- Model selection (ResNet18 vs ResNet50)
- Gradient accumulation for large models

## 🌐 API Deployment

### Production Deployment

For production deployment, consider:

1. **Environment Variables:**
   ```bash
   export MODEL_PATH=models/best_model.pth
   export MODEL_NAME=resnet18
   ```

2. **Docker Deployment:**
   ```dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

3. **Cloud Deployment:**
   - AWS EC2 / Lambda
   - Google Cloud Run
   - Azure Container Instances
   - Heroku

4. **Security Considerations:**
   - Rate limiting
   - Input validation
   - CORS configuration
   - Authentication/authorization

## 🔗 System Integration

### Frontend Integration

The web interface (`frontend/index.html`) provides:
- Drag-and-drop image upload
- Real-time API status checking
- Visual prediction results with confidence bars
- Top-3 predictions display
- Error handling and user feedback

### Mobile App Integration

The API can be integrated into mobile apps:

```python
# Example: Python requests
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/predict', 
                           files={'file': f})
    result = response.json()
```

```javascript
// Example: JavaScript fetch
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🤔 Ethical Considerations

### 1. Bias and Fairness

**Concerns:**
- Dataset may be biased toward certain cuisines or food types
- Cultural representation may be limited
- Geographic bias in training data

**Mitigation:**
- Use diverse datasets with global food representation
- Regularly audit model performance across different food categories
- Document dataset demographics and limitations
- Consider fairness metrics in evaluation

### 2. Privacy

**Concerns:**
- User-uploaded images may contain sensitive information
- Location metadata in images
- Personal dietary information

**Mitigation:**
- Implement image preprocessing to strip metadata
- Clear data retention policies
- Secure API endpoints (HTTPS)
- User consent for data usage
- Option to process images locally

### 3. Accuracy and Reliability

**Concerns:**
- Misclassification could affect user decisions
- Not suitable for medical or nutritional advice
- Model limitations in edge cases

**Mitigation:**
- Clear disclaimers about model limitations
- Confidence thresholds for predictions
- Human review for critical applications
- Regular model updates and retraining

### 4. Environmental Impact

**Concerns:**
- Energy consumption during training
- Carbon footprint of GPU usage

**Mitigation:**
- Use efficient model architectures
- Optimize training hyperparameters
- Consider cloud providers with renewable energy
- Model quantization for deployment

### 5. Accessibility

**Concerns:**
- Web interface may not be accessible to all users
- API may require technical knowledge

**Mitigation:**
- Ensure web interface follows WCAG guidelines
- Provide clear documentation
- Support multiple input methods
- Consider voice interfaces for accessibility

## 📝 Presentation Tips

When presenting this system:

1. **Demonstrate the Pipeline:**
   - Show dataset structure
   - Display training process and metrics
   - Demonstrate API endpoints
   - Show web interface in action

2. **Highlight Key Features:**
   - Transfer learning approach
   - Comprehensive evaluation metrics
   - Real-time prediction capability
   - Modern, user-friendly interface

3. **Discuss Challenges:**
   - Overfitting mitigation strategies
   - Class imbalance handling
   - Computational resource management

4. **Address Ethics:**
   - Bias considerations
   - Privacy measures
   - Accuracy limitations
   - Environmental impact

## 🛠️ Troubleshooting

### Model Not Loading
- Ensure model file exists at specified path
- Check model architecture matches saved model
- Verify class names are consistent

### API Connection Issues
- Check if API server is running
- Verify CORS settings
- Check firewall/port settings

### Training Errors
- Verify dataset structure is correct
- Check GPU availability (if using CUDA)
- Reduce batch size if out of memory
- Ensure sufficient disk space for checkpoints

## 📚 References

- PyTorch Documentation: https://pytorch.org/docs/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Food-101 Dataset: https://www.kaggle.com/datasets/dansbecker/food-101
- ResNet Paper: https://arxiv.org/abs/1512.03385

## 📄 License

This project is provided as-is for educational purposes.

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Note**: This system is designed for educational purposes. For production use, additional considerations for security, scalability, and reliability should be implemented.

