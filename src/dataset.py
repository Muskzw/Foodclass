"""
Dataset handling module for food classification.
Supports data loading, preprocessing, and augmentation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class FoodDataset(Dataset):
    """Custom dataset for food images."""
    
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir: Root directory containing class folders
            transform: Optional transform to be applied on a sample
            split: 'train', 'val', or 'test'
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Build class mapping and collect images
        if os.path.exists(self.root_dir):
            classes = sorted([d for d in os.listdir(self.root_dir) 
                            if os.path.isdir(os.path.join(self.root_dir, d))])
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
            
            for class_name in classes:
                class_dir = os.path.join(self.root_dir, class_name)
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms():
    """Get data transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_data_loaders(data_dir, batch_size=32, num_workers=2):
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing 'train', 'val', 'test' folders
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = FoodDataset(data_dir, transform=train_transform, split='train')
    val_dataset = FoodDataset(data_dir, transform=val_transform, split='val')
    test_dataset = FoodDataset(data_dir, transform=val_transform, split='test')
    
    # Get class names
    class_names = list(train_dataset.idx_to_class.values()) if len(train_dataset) > 0 else []
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    ) if len(train_dataset) > 0 else None
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    ) if len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    ) if len(test_dataset) > 0 else None
    
    return train_loader, val_loader, test_loader, class_names

