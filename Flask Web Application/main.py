from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from collections import deque

app = Flask("Fabric Inspection using Computer Vision")

# Load YOLO model (adjust the path as needed)
model_yolo = YOLO('/Users/randithasenarathne/FYP/runs/classify/train/weights/best.pt')

# Variables to handle pausing, fault rate calculation, and cooldown
is_paused = False
fault_points = 0
frame_count = 0
recent_defects = deque(maxlen=3)  # To track recent defects for long defects
defects_within_meter = deque(maxlen=30)  # To track defects within a meter
last_detected_class = None
cooldown_counter = 0
roll_length = 0
roll_width = 0
detected_defects = []  # List to store all detected defects for the report

# Define class names based on model predictions
class_names = ['foreign yarn', 'good', 'hole', 'slub', 'surface contamination']

# Preprocess image for YOLO model
def preprocess_image(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame.astype(np.float32) / 255.0
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)
    rgb_frame = np.concatenate([expanded_frame] * 3, axis=-1)
    rgb_frame = (rgb_frame * 255).astype(np.uint8)
    batch_frame = np.expand_dims(rgb_frame, axis=0)
    return batch_frame

# Post-process YOLO results
def post_process_results(results):
    if isinstance(results, list) and len(results) > 0:
        results = results[0]
    if hasattr(results, 'probs'):
        probs = results.probs.data.cpu().numpy()
        predicted_class = np.argmax(probs)
    else:
        print(results)
        raise ValueError("Unknown results structure")
    return predicted_class

# Function to calculate fault rate
def calculate_fault_rate(fault_points, roll_length, roll_width):
    return (fault_points * 0.84 * 100) / (roll_length * roll_width)

# Analyze defects for clustering or long defects
def analyze_defects(recent_defects, defects_within_meter, defect_type):
    fault_points = 0
    recent_defects.append(defect_type)
    defects_within_meter.append(defect_type)

    # Check if there are more than 3 defects within a meter
    if len(defects_within_meter) >= 3:
        defect_count = sum(1 for defect in defects_within_meter if defect in ['slub', 'foreign yarn', 'surface contamination'])
        if defect_count >= 3:
            fault_points += 4  # Assign 4 fault points for a cluster of defects detected
            cluster_message = "Cluster of defects detected"
        else:
            cluster_message = None
    else:
        cluster_message = None

    # Check for long defects (excluding holes)
    if defect_type in ['slub', 'foreign yarn', 'surface contamination'] and len(recent_defects) == recent_defects.maxlen:
        fault_points += 4  # Assign 4 fault points for a long defect detected
        long_defect_message = "Long defect detected"
    else:
        long_defect_message = None

    messages = [msg for msg in [cluster_message, long_defect_message] if msg]
    return messages, fault_points

# Process frame and predict fabric defect
def process_frame(frame, model, fault_points, recent_defects, cooldown_counter, last_detected_class, defects_within_meter):
    results = model.predict(Image.fromarray(frame))
    predicted_class = post_process_results(results)

    if predicted_class == last_detected_class:
        if cooldown_counter < 10:
            cooldown_counter += 1
            return fault_points, cooldown_counter, last_detected_class, class_names[predicted_class]
    else:
        cooldown_counter = 0
    last_detected_class = predicted_class

    # Analyze the defects for clustering or long defects
    defect_type = class_names[predicted_class]
    defect_analysis = analyze_defects(recent_defects, defects_within_meter, defect_type)

    if defect_analysis:
        print(defect_analysis)  # Log or display in the app

    # Store detected defect for the report
    detected_defects.append(defect_type)

    # Calculate fault points based on the defect type
    if defect_type in ['slub', 'foreign yarn', 'surface contamination']:
        fault_points += 1
    elif defect_type == 'hole':
        fault_points += 4

    return fault_points, cooldown_counter, last_detected_class, defect_type

# Video streaming route for live video
def generate_frames(model, roll_length, roll_width):
    global fault_points, cooldown_counter, last_detected_class, is_paused
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Check if paused
        if is_paused:
            continue

        fault_points, cooldown_counter, last_detected_class, defect_type = process_frame(
            frame, model, fault_points, recent_defects, cooldown_counter, last_detected_class, defects_within_meter
        )

        # Display defect type on the video feed
        cv2.putText(frame, f"Defect: {defect_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# Start the detection
@app.route('/start_detection', methods=['POST'])
def start_detection():
    global roll_length, roll_width, fault_points, detected_defects
    roll_length = float(request.form['length'])
    roll_width = float(request.form['width'])
    fault_points = 0  # Reset fault points for new inspection
    detected_defects = []  # Reset defects list for new inspection
    return jsonify(status='started')

# Video feed route
@app.route('/video_feed')
def video_feed():
    length = request.args.get('length')
    width = request.args.get('width')
    
    if not length or not width:
        return "Length and width parameters are required", 400
    
    return Response(generate_frames(model_yolo, float(length), float(width)), mimetype='multipart/x-mixed-replace; boundary=frame')

# Pause detection
@app.route('/pause', methods=['POST'])
def pause():
    global is_paused
    is_paused = True
    return jsonify({'status': 'paused'})

# Resume detection
@app.route('/resume', methods=['POST'])
def resume():
    global is_paused
    is_paused = False
    return jsonify({'status': 'resumed'})

# Stop detection and provide the report
@app.route('/stop', methods=['POST'])
def stop():
    global is_paused, fault_points, roll_length, roll_width
    is_paused = True
    fault_rate = calculate_fault_rate(fault_points, roll_length, roll_width)
    
    report = {
        'length': roll_length,
        'width': roll_width,
        'defects': detected_defects,
        'fault_rate': fault_rate
    }
    
    return jsonify({'status': 'stopped', 'report': report})

if __name__ == "__main__":
    app.run(debug=True)
