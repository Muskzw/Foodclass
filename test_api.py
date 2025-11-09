"""
Simple test script for the Food Classification API.
Tests the API endpoints to ensure everything is working.
"""

import requests
import os
import sys

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  Model loaded: {data.get('model_loaded', False)}")
            print(f"  Device: {data.get('device', 'unknown')}")
            print(f"  Classes: {data.get('num_classes', 0)}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_classes():
    """Test classes endpoint."""
    print("\nTesting /classes endpoint...")
    try:
        response = requests.get(f"{API_URL}/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Classes endpoint working")
            print(f"  Number of classes: {data.get('num_classes', 0)}")
            if data.get('classes'):
                print(f"  Sample classes: {data['classes'][:5]}")
            return True
        else:
            print(f"✗ Classes endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Classes endpoint failed: {e}")
        return False

def test_predict(image_path):
    """Test prediction endpoint."""
    print(f"\nTesting /predict endpoint with {image_path}...")
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Prediction successful")
            print(f"  Predicted class: {data.get('predicted_class', 'N/A')}")
            print(f"  Confidence: {data.get('confidence', 0):.2%}")
            if 'top_predictions' in data:
                print(f"  Top 3 predictions:")
                for pred in data['top_predictions']:
                    print(f"    - {pred['class']}: {pred['confidence']:.2%}")
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("Food Classification API Test Suite")
    print("=" * 50)
    
    # Check if API is running
    try:
        requests.get(f"{API_URL}/health", timeout=2)
    except:
        print(f"✗ Cannot connect to API at {API_URL}")
        print("  Please start the API server first:")
        print("  python scripts/run_api.py")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(test_health())
    results.append(test_classes())
    
    # Test prediction if image provided
    if len(sys.argv) > 1:
        results.append(test_predict(sys.argv[1]))
    else:
        print("\nSkipping prediction test (no image provided)")
        print("  Usage: python test_api.py <image_path>")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("=" * 50)

if __name__ == '__main__':
    main()

