import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'auth'

# --- DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    histories = db.relationship('History', backref='user', lazy=True)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    word = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
with app.app_context():
    db.create_all()

# --- AI & CAMERA SETUP ---
engine = pyttsx3.init()
engine.setProperty('rate', 150)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

print("Loading the AI brain...")
model = tf.keras.models.load_model("upgraded_model.h5")
gesture_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

state = {
    "current_sentence": "",
    "current_letter": "",
    "confidence": 0
}

engine_lock = threading.Lock()

def speak_text(text):
    with engine_lock:
        engine.say(text)
        engine.runAndWait()

def generate_frames():
    global state
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        current_letter_detected = ""
        conf = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                row_data = []
                for landmark in hand_landmarks.landmark:
                    row_data.append(landmark.x)
                    row_data.append(landmark.y)
                
                X_live = np.array([row_data])
                X_live_lstm = np.expand_dims(X_live, axis=1)
                
                predictions = model(X_live_lstm, training=False)
                predictions = predictions.numpy()
                predicted_index = np.argmax(predictions[0])
                confidence = predictions[0][predicted_index]
                
                if confidence > 0.8:
                    current_letter_detected = gesture_names[predicted_index]
                    conf = int(confidence * 100)

        state["current_letter"] = current_letter_detected
        state["confidence"] = conf
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# --- WEB ROUTES ---
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
            word_to_speak = state["current_sentence"]
            
            # --- SAVE TO DATABASE IF LOGGED IN ---
            if current_user.is_authenticated:
                new_history = History(user_id=current_user.id, word=word_to_speak)
                db.session.add(new_history)
                db.session.commit()
            
            threading.Thread(target=speak_text, args=(word_to_speak,)).start()
            state["current_sentence"] = "" 
            
    return jsonify({"success": True, "sentence": state["current_sentence"]})

# --- AUTH & HISTORY ROUTES ---
@app.route('/auth', methods=['GET', 'POST'])
def auth():
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')
        
        if action == 'login':
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for('history'))
            else:
                flash('Invalid username or password', 'error')
                
        elif action == 'signup':
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Username already exists', 'error')
            else:
                new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user)
                return redirect(url_for('history'))
                
    return render_template('auth.html')

@app.route('/history')
@login_required
def history():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    return render_template('history.html', history=user_history)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
