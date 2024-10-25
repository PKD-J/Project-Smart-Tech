import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt') 

# Perform prediction
res = model.predict(
    source='Aitin0942_jpg.rf.00ec9dee0c8acc633fb6679b0434fbba.jpg',
    conf=0.3,
)

# Load the image
image_path = 'Aitin0942_jpg.rf.00ec9dee0c8acc633fb6679b0434fbba.jpg'
image = cv2.imread(image_path)

class_names = {0: 'Safety-Harness', 1: 'Welding-Helmet',2: 'ear protection',3: 'glasses',4: 'gloves',5: 'helmet',6: 'mask',7: 'safety-shoes',8: 'vest'}

# Iterate over the results and draw bounding boxes
for result in res:
    boxes = result.boxes.xyxy  # box with xyxy format, (N, 4)
    confidences = result.boxes.conf  # confidence score, (N, 1)
    classes = result.boxes.cls  # class, (N, 1)

    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)  # Convert to integers
        class_id = int(cls)
        label = f'{class_names[class_id]}: {conf:.2f}'
        print(label)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

        # Put the label on the bounding box
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Display the image with bounding boxes

cv2.imshow('display', image)
cv2.waitKey(0)
cv2.destroyAllWindows() 
