"""
Training script for food classification model.
Includes training loop, validation, and evaluation metrics.
"""

import sys
import os

# Add parent directory to path to allow imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import get_model, save_model
from src.dataset import get_data_loaders


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model, train_loader, val_loader, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.writer = SummaryWriter('runs/food_classification')
        self.current_lr = lr
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, num_epochs, save_path='models'):
        """Train the model for multiple epochs."""
        os.makedirs(save_path, exist_ok=True)
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = 10
        
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # Learning rate scheduling
            old_lr = self.current_lr
            self.scheduler.step(val_loss)
            self.current_lr = self.optimizer.param_groups[0]['lr']
            if old_lr != self.current_lr:
                print(f"Learning rate reduced from {old_lr:.6f} to {self.current_lr:.6f}")
            
            # Save history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(save_path, 'best_model.pth')
                save_model(self.model, model_path, getattr(self, 'class_names', []))
                print(f"Saved best model with val_acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.writer.close()
        return self.train_losses, self.val_losses, self.train_accs, self.val_accs
    
    def evaluate_detailed(self, test_loader, class_names):
        """Detailed evaluation with metrics."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        from sklearn.metrics import classification_report
        print(classification_report(all_labels, all_preds, 
                                    target_names=class_names, zero_division=0))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("\nConfusion matrix saved to confusion_matrix.png")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Accuracy')
        ax2.plot(self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Training curves saved to {save_path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train food classification model')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing train/val/test folders')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet50', 'cnn'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models', help='Model save directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    
    if train_loader is None:
        print("Error: No training data found. Please ensure data directory structure is correct.")
        print("Expected structure: data/train/class1/, data/val/class1/, data/test/class1/")
        return
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    # Create model
    model = get_model(args.model, num_classes=len(class_names), pretrained=True)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device, lr=args.lr)
    trainer.class_names = class_names
    
    # Train
    trainer.train(args.epochs, save_path=args.save_dir)
    
    # Plot curves
    trainer.plot_training_curves()
    
    # Detailed evaluation on test set
    if test_loader is not None:
        trainer.evaluate_detailed(test_loader, class_names)


if __name__ == '__main__':
    main()

