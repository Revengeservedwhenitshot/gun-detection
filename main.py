import cv2
import torch
import sqlite3
import os
from datetime import datetime
from flask import Flask, render_template, Response
from plyer import notification  # Import the plyer library for desktop alerts

app = Flask(__name__)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (timestamp TEXT, object_type TEXT, image_path TEXT)''')
    conn.commit()
    conn.close()

# Log detection event
def log_detection(label, frame):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_path = f"images/{timestamp}.jpg"
    cv2.imwrite(image_path, frame)
    print(f"Image saved at: {image_path}")

    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("INSERT INTO detections (timestamp, object_type, image_path) VALUES (?, ?, ?)",
              (timestamp, label, image_path))
    conn.commit()
    conn.close()
    print(f"Detection logged: {timestamp}, {label}")

# Create images directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Initialize database
init_db()

# Load YOLOv5 model (ensure the correct path is used for the custom model)
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="D:/test/yolov5/runs/train/exp5/weights/best.pt")
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Initialize last alert time (global variable to track when the last alert was triggered)
last_alert_time = datetime.min  # Set to a very old date initially

@app.route('/')
def index():
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute("SELECT * FROM detections")
    logs = c.fetchall()
    conn.close()
    return render_template('index.html', logs=logs)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global last_alert_time  # Access the global variable for last alert time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No frame captured")
            break

        # Inference
        model.conf = 0.8  # Set confidence threshold
        try:
            results = model(frame)  # Run inference on the frame
            print("Inference completed successfully")
        except Exception as e:
            print(f"Error during inference: {e}")
            break

        # Render the results on the frame
        result_frame = results.render()[0]  # Get the frame with bounding boxes and labels

        # Check if any object of interest (knife or gun) is detected
        try:
            detections = results.pandas().xywh[0]  # Convert detections to pandas dataframe
            print("Detections processed successfully")
        except Exception as e:
            print(f"Error processing detections: {e}")
            break

        # Loop through detections to find "knife" or "gun"
        for index, row in detections.iterrows():
            label = row['name']  # Detected label (e.g., 'knife', 'gun')
            confidence = row['confidence']  # Confidence score

            # Check if a "knife" or "gun" is detected with high confidence
            if label in ['knife', 'gun'] and confidence >= 0.8:
                current_time = datetime.now()
                # Display desktop notification
                notification.notify(
                        title=f"Alert: {label.capitalize()} Detected!",
                        message=f"A {label} has been detected with {confidence * 100:.2f}% confidence.",
                        timeout=10  # Notification duration in seconds
                    )
                print(f"Alert: {label} detected with {confidence * 100:.2f}% confidence.")

                # Log the detection event
                log_detection(label, frame)

                # Update the last alert time
                last_alert_time = current_time

        # Display the frame
        ret, buffer = cv2.imencode('.jpg', result_frame)  # Corrected line
        if not ret:
            print("Error: Unable to encode frame.")
            continue  # Skip this frame if encoding fails

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
