from ultralytics import YOLO
from pathlib import Path

# === CONFIG ===
DATA_PATH = str(Path("output/data.yaml").resolve())  # your data.yaml
MODEL = "yolov8n.pt"  # lightweight, fast — or try yolov8s.pt if system allows

model = YOLO(MODEL)

# === OPTIMIZED TRAINING ===
model.train(
    data=DATA_PATH,
    epochs=40,                 # 🔁 More epochs = better convergence
    imgsz=416,                 # 📏 Standard resolution (can go to 416 for speed or 960 for precision)
    batch=4,                   # 🧠 Adjust based on RAM/VRAM (try 4 if CPU only, 16+ on GPU)
    patience=20,               # ⏳ Early stopping patience
    name="fabric_synth_v12",  # 📁 Output folder
    device="cpu",              # ⚙️ or 'cuda:0' if you have GPU
    optimizer="Adam",          # ⚡ Faster convergence on synthetic/small datasets
    augment=True,              # 🔄 Enable data augmentation
    verbose=True,
    single_cls=False,          # Use True if ALL defects are same class (else False for multi)
    cos_lr=True,               # 🎯 Smoother learning rate decay
    warmup_epochs=2,           # 🚀 Warm-up before full training
    lr0=0.001,                 # 🔧 Initial learning rate (lower = stable)
    lrf=0.01,                  # 🔧 Final learning rate
    hsv_h=0.015,               # 🎨 Color augmentation
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,               # 📐 Slight rotation
    translate=0.1,
    scale=0.5,
    fliplr=0.5,                # 🔁 Flip horizontally
    mosaic=1.0,                # 🧵 Great for synthetic data
    mixup=0.0,                 # 🎨 Add mixup if dataset is small
    dropout=0.1,               # 🚧 Avoid overfitting
)
