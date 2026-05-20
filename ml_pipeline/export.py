from ultralytics import YOLO

def export_model():
    # Let's load our trained model
    model = YOLO("runs/detect/train/weights/best.pt")

    print("--- Exporting model to optimized TFLite format with Float16 quantization ---")
    
    # Export to TFLite with Float16 quantization enabled for acceleration on mobile GPUs/NPUs
    exported_path = model.export(
        format="tflite",
        int8=False,      # Stick with float16 for better precision
        half=True,       # Enable FP16 quantization
        imgsz=320        # Reducing the input size to 320x320 for maximum FPS on mobile
    )
    
    print(f"--- Optimized model exported successfully: {exported_path} ---")

if __name__ == "__main__":
    export_model()