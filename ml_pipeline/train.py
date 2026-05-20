import os
from ultralytics import YOLO

def main():
    # 1. Load the pre-trained Nano YOLOv8 architecture
    model = YOLO("yolov8n.pt")

    # 2. Path to the dataset 
    dataset_path = "coco8.yaml" 

    print("--- Starting Transfer Learning Stage ---")
    # 3. Train model
    results = model.train(
        data=dataset_path,
        epochs=10,        # Number of epoch 
        imgsz=640,        # Standard image size for YOLO
        device="cpu"      # Change this to “cuda” or 0 if you're running in Google Colab with a GPU
    )
    
    print("--- Training completed. Model saved to runs/detect/train/weights/best.pt ---")

if __name__ == "__main__":
    main()