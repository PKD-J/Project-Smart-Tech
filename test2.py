import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(1)

class_names = {
    0: 'Safety-Harness',
    1: 'Welding-Helmet',
    2: 'ear protection',
    3: 'glasses',
    4: 'gloves',
    5: 'helmet',
    6: 'mask',
    7: 'safety-shoes',
    8: 'vest'
}

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:  # If there are no frames, stop
        break

    # Perform prediction
    res = model.predict(source=frame, conf=0.5)

    # Iterate over the results and draw bounding boxes
    for result in res:
        boxes = result.boxes.xyxy  # Box with xyxy format, (N, 4)
        confidences = result.boxes.conf  # Confidence score, (N, 1)
        classes = result.boxes.cls  # Class, (N, 1)

        for box, conf, cls in zip(boxes, confidences, classes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            class_id = int(cls)
            label = f'{class_names[class_id]}: {conf:.2f}'
            print(label)

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

            # Put the label on the bounding box
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display the frame with bounding boxes
    cv2.imshow('Webcam Display', frame)

    # Press 'q' to exit the webcam display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
