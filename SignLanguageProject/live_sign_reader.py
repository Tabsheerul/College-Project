import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3

# 1. Setup the Text-to-Speech Voice Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speed of speech

# 2. Setup the MediaPipe Hand Tracker
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 3. Load Your AI Brain
print("Loading the AI brain...")
model = tf.keras.models.load_model("upgraded_model.h5")
gesture_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# --- NEW WORD BUILDER VARIABLES ---
current_sentence = ""
current_letter = ""

# 4. Turn on the Camera
cap = cv2.VideoCapture(0)
print("\n" + "="*40)
print("🎥 CAMERA ON! Here are your controls:")
print("👉 SPACEBAR : Add the current letter to your word.")
print("👉 ENTER    : Speak the word out loud & clear the screen.")
print("👉 BACKSPACE: Delete the last letter.")
print("👉 ESC      : Close the application.")
print("="*40 + "\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Track the hand
    results = hands.process(rgb_frame)
    current_letter = "" # Reset the live letter every frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the 42 math coordinates
            row_data = []
            for landmark in hand_landmarks.landmark:
                row_data.append(landmark.x)
                row_data.append(landmark.y)
            
            X_live = np.array([row_data])
            
            # Predict the sign
            predictions = model.predict(X_live, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_index]
            
            # Show the live prediction at the top of the screen
            if confidence > 0.8:
                current_letter = gesture_names[predicted_index]
                cv2.putText(frame, f"Live Sign: {current_letter} ({int(confidence*100)}%)", 
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- NEW GUI: Draw the sentence bar at the bottom ---
    # Create a black rectangle at the bottom of the camera window
    cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1) 
    # Write the current sentence in the black box
    cv2.putText(frame, f"Word: {current_sentence}", (10, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Show the final window
    cv2.imshow("Final Year Project - Live Sign Reader", frame)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27: # ESC key to exit
        break
    elif key == 32: # SPACEBAR to add a letter
        if current_letter != "":
            current_sentence += current_letter
            print(f"Added: {current_letter}")
    elif key == 8: # BACKSPACE to delete the last letter
        current_sentence = current_sentence[:-1]
    elif key == 13: # ENTER key to speak!
        if current_sentence != "":
            print(f"Speaking: {current_sentence}")
            engine.say(current_sentence)
            engine.runAndWait() # This forces the voice to finish speaking before the camera continues
            current_sentence = "" # Clear the sentence for the next word

# Clean up
cap.release()
cv2.destroyAllWindows()