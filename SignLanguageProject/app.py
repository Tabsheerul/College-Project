from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading

app = Flask(__name__)

# --- AI & CAMERA SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("Loading the AI brain...")
model = tf.keras.models.load_model("upgraded_model.h5")

# UPDATE THIS LIST if you trained more letters! (e.g., A through G)
gesture_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

cap = cv2.VideoCapture(0)

# --- GLOBAL VARIABLES TO HOLD THE APP STATE ---
current_letter = ""
current_sentence = ""

# --- VIDEO GENERATOR (Feeds frames to HTML) ---
def generate_frames():
    global current_letter
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        current_letter = "" # Reset every frame

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                row_data = []
                for landmark in hand_landmarks.landmark:
                    row_data.append(landmark.x)
                    row_data.append(landmark.y)
                
                X_live = np.array([row_data])
                predictions = model(X_live, training=False).numpy()
                predicted_index = np.argmax(predictions[0])
                confidence = predictions[0][predicted_index]
                
                if confidence > 0.8:
                    current_letter = gesture_names[predicted_index]

        # Convert the frame to a format the web browser can display
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- FLASK WEB ROUTES ---

@app.route('/')
def index():
    # This serves the beautiful HTML page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # This continuously streams the camera to the HTML page
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add', methods=['POST'])
def add_letter():
    global current_sentence, current_letter
    if current_letter:
        current_sentence += current_letter
    return jsonify({'sentence': current_sentence})

@app.route('/delete', methods=['POST'])
def delete_letter():
    global current_sentence
    current_sentence = current_sentence[:-1]
    return jsonify({'sentence': current_sentence})

@app.route('/clear', methods=['POST'])
def clear_sentence():
    global current_sentence
    current_sentence = ""
    return jsonify({'sentence': current_sentence})

@app.route('/speak', methods=['POST'])
def speak_sentence():
    global current_sentence
    if current_sentence:
        # We initialize pyttsx3 exactly when we need it so it plays nicely with the web server!
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(current_sentence)
        engine.runAndWait()
        current_sentence = "" # Auto-clear after speaking
    return jsonify({'sentence': current_sentence})

@app.route('/get_live_sign')
def get_live_sign():
    # The webpage will constantly call this to see what the AI is looking at
    global current_letter
    return jsonify({'letter': current_letter})

if __name__ == '__main__':
    # Starts the web server!
    app.run(debug=True, threaded=True)