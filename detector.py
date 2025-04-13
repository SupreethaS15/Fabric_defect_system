# detector.py

from ultralytics import YOLO
import cv2
import numpy as np
import os

CLASS_LABELS = {
    0: "Tear",
    1: "Unstitched",
    2: "Hole"
}

def load_model(model_path):
    """
    Dynamically loads a YOLOv8 model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[❌] Model not found at: {model_path}")
    return YOLO(model_path)


def detect_damage(image, model, conf_threshold=0.3):
    """
    Runs YOLOv8 detection on input image using given model.
    Returns annotated image and detection results.
    """
    results_list = []

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(rgb_img, verbose=False)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        conf = float(conf)
        cls = int(cls)

        if conf >= conf_threshold:
            label = CLASS_LABELS.get(cls, "Unknown")
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            results_list.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)]
            })

    return image, results_list


def estimate_position_cm(bbox_center, dpi=300, fabric_speed_cm_s=10, frame_index=0, fps=30):
    time_sec = frame_index / fps
    distance_cm = fabric_speed_cm_s * time_sec
    return round(distance_cm, 2)


# Optional: main test block for image detection
if __name__ == "__main__":
    test_img = cv2.imread("test.jpg")
    model = load_model("models/old_best.pt")
    if test_img is not None:
        annotated, detections = detect_damage(test_img, model)
        print("[✅] Detections:", detections)
        cv2.imshow("YOLOv8 Fabric Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
