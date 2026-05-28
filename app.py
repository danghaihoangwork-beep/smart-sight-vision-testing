import cv2
import random
import numpy as np
import time
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO

app = Flask(__name__)

# --- 1. INITIALIZE YOLO & OPENCV CORE ---
try:
    model = YOLO('best.pt')
    print("🚀 [SYSTEM] Successfully loaded custom YOLO model (best.pt).")
except Exception as e:
    print(f"⚠️ [WARNING] Custom model not found. Defaulting to yolov8n.pt. Error: {e}")
    model = YOLO('yolov8n.pt')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- 2. GLOBAL CLINICAL STATE MANAGEMENT ---
DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
VISION_LEVELS = {0: "20/200", 1: "20/100", 2: "20/70", 3: "20/50", 4: "20/40", 5: "20/30", 6: "20/25", 7: "20/20"}
MAX_LEVEL = len(VISION_LEVELS) - 1

KNOWN_FACE_WIDTH = 14.3  
FOCAL_LENGTH = 600.0     
MIN_DIST, MAX_DIST = 50.0, 70.0

state = {
    'is_active': False,
    'test_finished': False,
    'max_attempts': 15,
    'target': random.choice(DIRECTIONS),
    'score': 0,
    'attempts': 0,       
    'time_left': 15,     
    'distance_cm': 60.0,
    'privacy_mode': False,
    'target_start_time': 0,
    'last_response_time': 0.0,
    'hesitation': False,
    'alert': "SYSTEM READY",
    'missing_face_frames': 0,
    'head_stability_pct': 100.0,
    'ambient_lux': 300,
    'fps': 30.0,
    'inference_ms': 12
}

prediction_history = deque(maxlen=15)
head_history = deque(maxlen=15)
CONFIDENCE_THRESHOLD = 0.50  # Hạ xuống 0.50 để AI nhạy hơn khi bạn giơ tay nhanh
prev_frame_time = time.time()

def calculate_lux(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return int(mean_brightness * 2.5) 

def generate_frames():
    global state, prediction_history, head_history, prev_frame_time
    cap = cv2.VideoCapture(0)
    smoothed_width = None

    while True:
        success, frame = cap.read()
        if not success: 
            break
        
        frame = cv2.flip(frame, 1)
        h_img, w_img, _ = frame.shape
        
        # --- MEASURE FPS ---
        curr_time = time.time()
        fps_val = 1.0 / (curr_time - prev_frame_time) if (curr_time - prev_frame_time) > 0 else 30.0
        prev_frame_time = curr_time
        state['fps'] = round((state['fps'] * 0.9) + (fps_val * 0.1), 1)

        # --- MEASURE AMBIENT LUX ---
        state['ambient_lux'] = calculate_lux(frame)

        # --- OPTIMIZED FACE TRACKING (Downsampled to 320x240 for 10x speed boost) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (320, 240))
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        is_valid_distance = False

        if len(faces) == 0:
            state['missing_face_frames'] += 1
            if state['missing_face_frames'] > 10: 
                state['alert'] = "ANTI-CHEAT: FACE NOT DETECTED"
        else:
            state['missing_face_frames'] = 0
            # Get largest face and upscale coordinates back to original size
            (x_small, y_small, w_small, h_small) = max(faces, key=lambda b: b[2] * b[3])
            scale_x = w_img / 320
            scale_y = h_img / 240
            x, y = int(x_small * scale_x), int(y_small * scale_y)
            w, h_face = int(w_small * scale_x), int(h_small * scale_y)
            
            if smoothed_width is None: 
                smoothed_width = w
            else: 
                smoothed_width = (smoothed_width * 0.88) + (w * 0.12)
            
            raw_dist = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / smoothed_width if smoothed_width > 0 else 60
            state['distance_cm'] = round((state['distance_cm'] * 0.7) + (raw_dist * 0.3), 1)

            # Head Stability
            cx, cy = x + w // 2, y + h_face // 2
            head_history.append((cx, cy))
            if len(head_history) == 15:
                std_x = np.std([pt[0] for pt in head_history])
                std_y = np.std([pt[1] for pt in head_history])
                jitter = (std_x + std_y) * 1000 / w_img
                state['head_stability_pct'] = round(max(0.0, min(100.0, 100.0 - jitter)), 1)

            if state['distance_cm'] < MIN_DIST:
                state['alert'] = "TOO CLOSE"
            elif state['distance_cm'] > MAX_DIST:
                state['alert'] = "TOO FAR"
            else:
                is_valid_distance = True
                if state['is_active'] and not state['test_finished']:
                    state['alert'] = "TRACKING ACTIVE"

            # Privacy Blur
            if state['privacy_mode']:
                try:
                    margin = 20
                    y1, y2 = max(0, y-margin), min(h_img, y+h_face+margin)
                    x1, x2 = max(0, x-margin), min(w_img, x+w+margin)
                    face_roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (99, 99), 25)
                except: 
                    pass

        # --- TIMER & SCORING LOGIC ---
        if state['is_active'] and not state['test_finished']:
            if not is_valid_distance or len(faces) == 0:
                state['target_start_time'] = time.time() - (15 - state['time_left'])
            else:
                elapsed = time.time() - state['target_start_time']
                state['time_left'] = max(0, 15 - int(elapsed))
                
                if state['time_left'] == 0:
                    state['attempts'] += 1
                    if state['attempts'] >= state['max_attempts']:
                        state['test_finished'] = True
                        state['alert'] = "ASSESSMENT COMPLETE"
                    else:
                        state['target'] = random.choice(DIRECTIONS)
                        state['target_start_time'] = time.time()
                        prediction_history.clear()

            # --- OPTIMIZED YOLO INFERENCE (Using imgsz=320 for lightning-fast speed) ---
            if is_valid_distance:
                inf_start = time.time()
                results = model(frame, imgsz=320, verbose=False)
                state['inference_ms'] = int((time.time() - inf_start) * 1000)
                
                detected_direction = None
                if len(results[0].boxes) > 0:
                    best_box = results[0].boxes[0]
                    if float(best_box.conf[0]) > CONFIDENCE_THRESHOLD:
                        try: 
                            detected_direction = model.names[int(best_box.cls[0])].strip().upper()
                        except: 
                            pass
                        
                        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (238, 211, 34), 2)
                        cv2.putText(frame, f"{detected_direction} {float(best_box.conf[0]):.2f}", 
                                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (238, 211, 34), 2)

                if detected_direction: 
                    prediction_history.append(detected_direction)
                
                correct_votes = prediction_history.count(state['target'])
                if correct_votes >= 10: 
                    rt = time.time() - state['target_start_time']
                    state['last_response_time'] = round(rt, 2)
                    state['hesitation'] = rt > 2.5 
                    state['score'] += 1
                    state['attempts'] += 1
                    
                    if state['attempts'] >= state['max_attempts']:
                        state['test_finished'] = True
                        state['alert'] = "ASSESSMENT COMPLETE"
                    else:
                        state['target'] = random.choice(DIRECTIONS)
                        state['target_start_time'] = time.time()
                        prediction_history.clear()

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_test():
    data = request.json or {}
    state['max_attempts'] = int(data.get('max_attempts', 15))
    state['score'] = 0
    state['attempts'] = 0
    state['target'] = random.choice(DIRECTIONS)
    state['target_start_time'] = time.time()
    state['hesitation'] = False
    state['last_response_time'] = 0.0
    state['test_finished'] = False
    state['is_active'] = True
    prediction_history.clear()
    return jsonify({'status': 'started'})

@app.route('/toggle_privacy', methods=['POST'])
def toggle_privacy():
    state['privacy_mode'] = not state['privacy_mode']
    return jsonify({'privacy_mode': state['privacy_mode']})

@app.route('/reset', methods=['POST'])
def reset_session():
    state['is_active'] = False
    state['test_finished'] = False
    state['score'] = 0
    state['attempts'] = 0
    state['time_left'] = 15
    state['alert'] = "SYSTEM READY"
    return jsonify({'status': 'reset'})

@app.route('/test_status')
def test_status():
    return jsonify({
        'is_active': state['is_active'],
        'test_finished': state['test_finished'],
        'max_attempts': state['max_attempts'],
        'target_direction': state['target'],
        'score': state['score'],
        'attempts': state['attempts'],
        'time_left': state['time_left'],
        'distance_cm': state['distance_cm'],
        'vision_level': VISION_LEVELS.get(min(state['score'], MAX_LEVEL), "20/20"),
        'response_time': state['last_response_time'],
        'hesitation': state['hesitation'],
        'alert': state['alert'],
        'privacy_mode': state['privacy_mode'],
        'head_stability_pct': state['head_stability_pct'],
        'ambient_lux': state['ambient_lux'],
        'fps': state['fps'],
        'inference_ms': state['inference_ms']
    })

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5050, debug=False)