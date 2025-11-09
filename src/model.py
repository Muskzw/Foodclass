"""
Model architecture definitions for food classification.
Supports CNN and ResNet architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Simple CNN architecture for food classification."""
    
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name='resnet18', num_classes=10, pretrained=True):
    """
    Get a model architecture.
    
    Args:
        model_name: 'resnet18', 'resnet50', or 'cnn'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for ResNet)
    
    Returns:
        Model instance
    """
    if model_name == 'cnn':
        model = SimpleCNN(num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def save_model(model, path, class_names):
    """Save model and metadata."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'model_architecture': model.__class__.__name__
    }, path)
    print(f"Model saved to {path}")


def load_model(path, model_name='resnet18', device='cpu'):
    """Load model and metadata."""
    checkpoint = torch.load(path, map_location=device)
    class_names = checkpoint['class_names']
    
    model = get_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, class_names

