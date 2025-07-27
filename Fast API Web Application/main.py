from fastapi import FastAPI, WebSocket, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from collections import deque
import torch
import asyncio
import base64

app = FastAPI(title="Fabric Inspection using Computer Vision")
templates = Jinja2Templates(directory=".")

# Check if CUDA or MPS is available
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load YOLO model onto the selected device
model_yolo = YOLO('best.pt').to(device)

# Variables to handle processing state
is_paused = False
is_stopped = False
fault_points = 0
recent_defects = deque(maxlen=30)  # Track 30 frames for 1 meter of fabric
detected_defects = []
cooldown_counter = 0
last_detected_class = None
class_names = ['foreign yarn', 'good', 'hole', 'slub', 'surface contamination']

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/video_feed")
async def websocket_video_feed(websocket: WebSocket):
    global is_paused, is_stopped, fault_points, cooldown_counter, last_detected_class
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    try:
        while not is_stopped:
            if is_paused:
                await asyncio.sleep(0.1)
                continue

            success, frame = cap.read()
            if not success:
                break

            # Preprocess the frame before sending to the model
            preprocessed_frame = preprocess_image(frame)

            # Process preprocessed frame for defect detection
            fault_points, cooldown_counter, last_detected_class, defect_type = process_frame(
                preprocessed_frame, model_yolo, fault_points, recent_defects, cooldown_counter, last_detected_class
            )

            # Encode original frame (not preprocessed) as JPEG, convert to base64 for transmission
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')

            # Send encoded frame and defect type over WebSocket
            try:
                await websocket.send_json({
                    "frame": frame_data,
                    "fault_points": fault_points,
                    "detected_defects": list(detected_defects),
                    "defect_type": defect_type,
                    "cooldown_counter": cooldown_counter
                })
            except RuntimeError as send_error:
                print(f"WebSocket send error: {send_error}")
                break  # Exit the loop if sending fails

            # Control frame rate
            await asyncio.sleep(0.03)  # ~30 FPS

    except Exception as e:
        print(f"WebSocket connection error: {e}")

    finally:
        cap.release()
        # Check if WebSocket is still open before trying to close it
        if not websocket.client_state.name == 'DISCONNECTED':
            try:
                await websocket.close()
            except RuntimeError:
                print("WebSocket was already closed.")

def preprocess_image(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to 64x64 if it's not already
    resized_frame = cv2.resize(gray_frame, (64, 64))

    # Normalize the frame 
    normalized_frame = resized_frame.astype(np.float32) / 255.0

    # Expand dimensions to add channel for grayscale (1 channel)
    expanded_frame = np.expand_dims(normalized_frame, axis=-1)

    # Convert grayscale to 3 channels by duplicating the grayscale channel
    rgb_frame = np.concatenate([expanded_frame] * 3, axis=-1)

    # Convert to 8-bit unsigned integer (0-255) format required by PIL
    rgb_frame = (rgb_frame * 255).astype(np.uint8)

    # Add batch dimension 
    batch_frame = np.expand_dims(rgb_frame, axis=0)

    return batch_frame

def process_frame(frame, model, fault_points, recent_defects, cooldown_counter, last_detected_class):
    # Convert frame to compatible format for model
    frame = Image.fromarray(frame.squeeze().astype(np.uint8), 'RGB')  # Removing the batch dimension and converting back to RGB
    results = model.predict(frame, device=device)
    predicted_class = post_process_results(results)

    # Handle cooldown logic for defect detection
    if predicted_class == last_detected_class:
        if cooldown_counter < 10:
            cooldown_counter += 1
            return fault_points, cooldown_counter, last_detected_class, class_names[predicted_class]
    else:
        cooldown_counter = 0
    last_detected_class = predicted_class

    defect_type = class_names[predicted_class]
    
    # Only append defects, excluding "good" predictions
    if defect_type != 'good':
        detected_defects.append(defect_type)

    # Adjust fault points based on defect type and additional conditions
    if defect_type in ['slub', 'foreign yarn', 'surface contamination']:
        recent_defects.append(defect_type)
        one_meter_defects = sum(1 for d in recent_defects if d in ['slub', 'foreign yarn', 'surface contamination'])

        if one_meter_defects >= 3:
            fault_points += 3
        elif one_meter_defects >= 2:
            fault_points += 2
    elif defect_type == 'hole':
        fault_points += 4

    return fault_points, cooldown_counter, last_detected_class, defect_type

def post_process_results(results):
    if isinstance(results, list) and len(results) > 0:
        results = results[0]
    if hasattr(results, 'probs'):
        probs = results.probs.data.cpu().numpy()
        predicted_class = np.argmax(probs)
    return predicted_class

@app.post("/start_detection")
async def start_detection():
    global fault_points, detected_defects, is_stopped
    fault_points = 0
    detected_defects = []
    is_stopped = False
    return JSONResponse(content={"status": "started"})

@app.post("/pause")
async def pause():
    global is_paused
    is_paused = True
    return JSONResponse(content={'status': 'paused'})

@app.post("/resume")
async def resume():
    global is_paused
    is_paused = False
    return JSONResponse(content={'status': 'resumed'})

@app.post("/stop")
async def stop():
    global is_paused, is_stopped
    is_paused = True
    is_stopped = True
    return JSONResponse(content={'status': 'stopped'})

@app.post("/generate_report")
async def generate_report(length: float = Form(...), width: float = Form(...)):
    global fault_points, detected_defects
    filtered_defects = [defect for defect in detected_defects if defect != 'good']
    fault_rate = (fault_points * 0.84 * 100) / (length * width) if length and width else 0
    report = {
        'length': length,
        'width': width,
        'defects': filtered_defects,
        'fault_rate': fault_rate
    }
    fault_points = 0
    detected_defects = []
    return JSONResponse(content={'status': 'report_generated', 'report': report})

