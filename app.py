import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# 1. Setup the Text-to-Speech Voice Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 2. Setup the MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 3. Load Your AI Brain
print("Loading the AI brain...")
model = tf.keras.models.load_model("upgraded_model.h5")
gesture_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# State variables
state = {
    "current_sentence": "",
    "current_letter": "",
    "confidence": 0
}

# We need a lock for the pyttsx3 engine since it runs in a separate thread
engine_lock = threading.Lock()

def speak_text(text):
    with engine_lock:
        engine.say(text)
        engine.runAndWait()

def generate_frames():
    global state
    cap = cv2.VideoCapture(0)
    
    # --- PERFORMANCE OPTIMIZATION 1: Lower resolution to speed up processing ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Track the hand
        results = hands.process(rgb_frame)
        current_letter_detected = ""
        conf = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract the 42 math coordinates
                row_data = []
                for landmark in hand_landmarks.landmark:
                    row_data.append(landmark.x)
                    row_data.append(landmark.y)
                
                X_live = np.array([row_data])
                
                # Reshape for LSTM
                X_live_lstm = np.expand_dims(X_live, axis=1)
                
                # --- PERFORMANCE OPTIMIZATION 2: Use direct call instead of .predict() ---
                # model.predict() is extremely slow for single frames because it's built for huge batches.
                predictions = model(X_live_lstm, training=False)
                predictions = predictions.numpy() # Convert from tensor back to normal array
                
                predicted_index = np.argmax(predictions[0])
                confidence = predictions[0][predicted_index]
                
                if confidence > 0.8:
                    current_letter_detected = gesture_names[predicted_index]
                    conf = int(confidence * 100)

        # Update global state for the frontend to fetch
        state["current_letter"] = current_letter_detected
        state["confidence"] = conf
        
        # --- PERFORMANCE OPTIMIZATION 3: Compress JPEG slightly for faster streaming ---
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    return jsonify(state)

@app.route('/action', methods=['POST'])
def handle_action():
    global state
    action = request.json.get('action')
    
    if action == 'add':
        if state["current_letter"]:
            state["current_sentence"] += state["current_letter"]
    elif action == 'delete':
        state["current_sentence"] = state["current_sentence"][:-1]
    elif action == 'speak':
        if state["current_sentence"]:
            # Run speaking in a background thread so it doesn't freeze the video feed
            threading.Thread(target=speak_text, args=(state["current_sentence"],)).start()
            state["current_sentence"] = "" # Clear after speaking
            
    return jsonify({"success": True, "sentence": state["current_sentence"]})

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
