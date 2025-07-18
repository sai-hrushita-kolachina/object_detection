### Object Detection in Real-Time with YOLOv8 ###

from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model (YOLOv8n is the nano version, fast and light)
model = YOLO("yolov8n.pt")  

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, stream=True)

    # Draw boxes and labels
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]                     # Confidence
            cls = int(box.cls[0])                  # Class index
            label = model.names[cls]               # Class name

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Output
    cv2.imshow("YOLOv8 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
