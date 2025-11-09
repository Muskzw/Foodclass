"""
Convenience script to run the FastAPI server.
"""

import uvicorn
import os
import sys

if __name__ == '__main__':
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check if model exists
    model_path = os.getenv('MODEL_PATH', 'models/best_model.pth')
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Please train a model first using: python src/train.py")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

