# Personal Protective Equipment (PPE) Detection Using YOLOv5

This project automates the detection of essential Personal Protective Equipment (PPE) items such as helmets, gloves, vests, shoes, and faces using a YOLOv5-based object detection model.

## 📌 Objectives

- Understand object detection and YOLO algorithms.
- Train a custom YOLOv5 model on a PPE dataset.
- Detect 5 PPE classes: **helmet**, **gloves**, **face**, **vest**, and **shoes**.
- Perform inference on images and videos.
- Evaluate using metrics like **mAP**, **F1-score**, **precision**, **recall**, and **loss**.

## 📁 Dataset Preparation

- **Source:** [Roboflow](https://roboflow.com/)
- **Format:** YOLOv5 (.txt) — `[class_id, x_center, y_center, width, height]`
- **Classes:**
  - Helmet
  - Gloves
  - Face
  - Vest
  - Shoes
- **Images:** ~1500–1600

## ⚙️ Methodology

### 1. Preprocessing

- Images resized
- YOLOv5 format annotations

### 2. Model Training

- **Model:** YOLOv5s
- **Framework:** PyTorch on Google Colab
- **Epochs:** 50
- **Batch size:** 16
- **Image size:** 640×640
- **Pretrained Weights:** `yolov5s.pt`, `yolov5m.pt`

### 3. Evaluation Metrics

| Metric           | Value (Approx.) | Comments                            |
|------------------|------------------|-------------------------------------|
| Train Loss       | ~0.02–0.03       | Low — indicates good learning       |
| Validation Loss  | ~0.02–0.043      | Low — good generalization           |
| Precision        | ~0.85–0.90       | High — most predictions correct     |
| Recall           | ~0.65–0.70       | Moderate — can improve              |
| mAP@0.5          | ~0.65            | Good detection accuracy             |
| mAP@0.5:0.95     | ~0.33            | Acceptable, room for improvement    |

## 🛠 Problems & Fixes

- Roboflow-exported `data.yaml` needed remapping to correct class labels.
- Class label misdetection required multiple model retrainings.

## ✅ Results

- Accurately detects all five PPE categories.
- Real-time inference supported on GPU systems.
- Outputs include bounding boxes, labels, and confidence scores.
  <img width="640" height="640" alt="image" src="https://github.com/user-attachments/assets/2f2d4509-2424-4aad-9e22-720cebedec0c" />


## 🔮 Future Scope

- CCTV integration with alert systems.
- Add more PPE types (e.g., goggles, earmuffs).
- Use larger, more diverse datasets.

## 🧪 Inference Script

Run inference on images or videos using the script below:

```python
import torch
from pathlib import Path
import cv2
import argparse

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.3

def run_inference(source):
    if source.endswith(('.jpg', '.png')):
        img = cv2.imread(source)
        results = model(img)
        results.render()
        cv2.imshow("PPE Detection", results.ims[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif source.endswith(('.mp4', '.avi')):
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            results.render()
            cv2.imshow("PPE Detection", results.ims[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unsupported file format")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path to image or video')
    args = parser.parse_args()
    run_inference(args.source)
