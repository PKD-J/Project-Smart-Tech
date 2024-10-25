from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import datetime
import os
import sqlite3
import time

app = Flask(__name__)
model = YOLO('best.pt')

# Define class names
class_names = {
    0: 'Safety-Harness',
    1: 'Welding-Helmet',
    2: 'Ear Protection',
    3: 'Glasses',
    4: 'Gloves',
    5: 'Helmet',
    6: 'Mask',
    7: 'Safety Shoes',
    8: 'Vest'
}

# Define classes for each option
class_options = {
    'option1': [1, 7, 4],  # Safety Shoes, Welding-Helmet, Gloves
    'option2': [5, 7, 3, 4, 8, 2, 6],  # Helmet, Safety Shoes, Glasses, Gloves, Vest, Ear Protection, Mask
    'option3': [5, 7, 3, 4, 8, 2, 0, 6]   # Helmet, Safety Shoes, Glasses, Gloves, Vest, Ear Protection, Safety-Harness, Mask
}

selected_classes = []
dangerous = False
last_dangerous_time = 0  # Track the last time a dangerous condition was logged
last_safe_time = 0  # Track the last time a safe condition was logged
dangerous_interval = 20  # Interval for logging dangerous events in seconds
safe_interval = 20  # Interval for logging safe events in seconds
dangerous_image_path = None  # Path to the dangerous image
safe_image_path = None  # Path to the safe image
danger_status_message = ""  # Variable to store the danger status message

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('detection_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            status TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_detection_data(status, image_path=None):
    conn = sqlite3.connect('detection_data.db')
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO detection_logs (timestamp, status, image_path)
        VALUES (?, ?, ?)
    ''', (timestamp, status, image_path))
    conn.commit()
    conn.close()

def get_all_detection_data():
    conn = sqlite3.connect('detection_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM detection_logs ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def generate_frames(selected_classes):
    global dangerous, last_dangerous_time, last_safe_time, dangerous_image_path, safe_image_path, danger_status_message
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform prediction
        res = model.predict(source=frame, conf=0.5)

        detected_classes = []

        # Draw bounding boxes
        for result in res:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confidences, classes):
                class_id = int(cls)

                if class_id in selected_classes:  # Check if class is selected
                    detected_classes.append(class_names[class_id])
                    x1, y1, x2, y2 = map(int, box)
                    label = f'{class_names[class_id]}: {conf:.2f}'

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Check for dangerous situation
        if not detected_classes:
            if time.time() - last_dangerous_time >= dangerous_interval:
                dangerous = True
                now = datetime.datetime.now()
                dangerous_image_path = f"static/dangerous_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(dangerous_image_path, frame)  # Save the dangerous image

                # Save detection data when a dangerous situation is detected
                save_detection_data('dangerous', dangerous_image_path)

                # Update the last dangerous time and status message
                last_dangerous_time = time.time()
                danger_status_message = "🚨 การตรวจจับอันตราย! 🚨"
        else:
            dangerous = False
            # Save detection data when a safe condition is detected
            if time.time() - last_safe_time >= safe_interval:
                safe_image_path = f"static/safe_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(safe_image_path, frame)  # Save the safe image
                save_detection_data('safe', safe_image_path)  # Save the status as safe in the database
                last_safe_time = time.time()
                
                danger_status_message = "สถานะ: ปลอดภัย"

        # Get current time for display
        now = datetime.datetime.now()
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # Put the current time on the frame
        cv2.putText(frame, current_time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the frame in JPEG format
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    detection_logs = get_all_detection_data()  # Get all logs from the database
    return render_template('index.html', selected_model=None, dangerous=False, dangerous_image_path=None, danger_status_message="", detection_logs=detection_logs)

@app.route('/select_classes', methods=['GET', 'POST'])
def select_classes():
    global selected_classes
    global dangerous_image_path
    global safe_image_path
    global danger_status_message
    selected_model = None
    if request.method == 'POST':
        option = request.form.get('class_option')
        selected_classes = class_options.get(option, [])
        # Set the selected model name based on the option
        if option == 'option1':
            selected_model = "ห้องเชื่อม"
        elif option == 'option2':
            selected_model = "ห้องเครื่องจักร"
        elif option == 'option3':
            selected_model = "ที่สูง"

        return render_template('index.html', selected_model=selected_model, dangerous=dangerous, dangerous_image_path=dangerous_image_path, safe_image_path=safe_image_path, danger_status_message=danger_status_message, detection_logs=get_all_detection_data())

    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(selected_classes), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/danger_status')
def danger_status():
    return jsonify({
        'dangerous': dangerous,
        'danger_message': danger_status_message,
        'dangerous_image_path': dangerous_image_path,
        'safe_image_path': safe_image_path
    })

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(host='0.0.0.0', port=5000, debug=True)
