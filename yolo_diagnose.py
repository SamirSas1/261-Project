import sys
import torch
from pathlib import Path

# === Configuration ===
MODEL_PATH = 'yolov5/best_yolo_model.pt'
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else 'test_image.jpg'  # Default image

# === Load YOLOv5 Model ===
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# === Run Inference ===
results = model(IMAGE_PATH)

# === Print + Save Results ===
results.print()                     # Show results in console
results.show()                      # Open image with boxes
results.save(save_dir='output')    # Save image with boxes

# === Optional: Print box coords
for *box, conf, cls in results.xyxy[0]:
    print(f"\n🧠 Tumor Detected with {conf:.2f} confidence")
    print(f"📦 Bounding Box: {box}")
