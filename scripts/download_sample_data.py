"""
Script to download and prepare a sample food dataset.
This script helps set up a small dataset for testing the system.
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil

def download_file(url, filename):
    """Download a file from URL."""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
    print(f"\nDownloaded {filename}")

def setup_sample_structure():
    """Create sample directory structure."""
    data_dir = Path('data')
    splits = ['train', 'val', 'test']
    
    # Sample food classes
    classes = ['pizza', 'burger', 'sushi', 'pasta', 'salad']
    
    for split in splits:
        for cls in classes:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {data_dir}")
    print("Please add images to the respective class folders:")
    for split in splits:
        for cls in classes:
            print(f"  - {data_dir}/{split}/{cls}/")

if __name__ == '__main__':
    print("=" * 50)
    print("Food Classification Dataset Setup")
    print("=" * 50)
    print("\nThis script will help you set up the dataset structure.")
    print("\nFor a full dataset, please download Food-101 from Kaggle:")
    print("  kaggle datasets download -d datatang/datatang-food101")
    print("\nOr use Food-11:")
    print("  kaggle datasets download -d trolukovich/food11-image-dataset")
    print("\nCreating sample directory structure...")
    
    setup_sample_structure()
    
    print("\n" + "=" * 50)
    print("Next steps:")
    print("1. Download a food dataset from Kaggle")
    print("2. Extract and organize into train/val/test folders")
    print("3. Each folder should contain class subfolders with images")
    print("4. Run training: python src/train.py --data_dir data")
    print("=" * 50)

