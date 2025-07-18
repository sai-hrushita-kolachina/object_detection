# Real-Time Object Detection using YOLOv8 and OpenCV

This project demonstrates real-time object detection using **YOLOv8 (You Only Look Once)** and **OpenCV**. It leverages the `ultralytics` library for seamless integration of YOLOv8 models with webcam feeds, enabling object recognition and labeling in real-time.

---

##  Technologies Used
- Python 
- [YOLOv8](https://github.com/ultralytics/ultralytics) (`ultralytics` package)
- OpenCV (`cv2`)

---

## Features
- Real-time object detection using webcam feed
- Uses YOLOv8-nano (`yolov8n.pt`) for lightweight, fast inference
- Displays object labels and confidence scores
- Draws bounding boxes around detected objects
- Press **Q** to exit the application

---

## How to Run the Project

### Clone the Repository
```bash
git clone https://github.com/sai-hrushita-kolachina/object_detection.git
cd yolov8-object-detection
pip install ultralytics opencv-python
pyhton objectdetection.py
