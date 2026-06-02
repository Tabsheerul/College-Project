import cv2
import mediapipe as mp
import pandas as pd
import sys

DATA_FILE = "gesture_dataset.csv"

# 1. Ask user for the letter
letter = input("Enter the letter you want to train (A-Y): ").upper()
if len(letter) != 1 or not letter.isalpha():
    print("Invalid input. Please enter a single letter from A to Y.")
    sys.exit()

try:
    num_frames_to_collect = int(input("How many frames do you want to record? (Recommended: 300): "))
except ValueError:
    print("Please enter a valid number.")
    sys.exit()

# 2. Setup MediaPipe Camera
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
print(f"\n--- Get ready for letter '{letter}' ---")
print("Make sure the camera window is selected, then press 's' on your keyboard to start recording.")
print("Press 'ESC' at any time to cancel.")

recording = False
frames_collected = 0
data_to_append = []

while True:
    success, frame = cap.read()
    if not success:
        print("Camera error.")
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Only save data if we are currently recording
            if recording:
                row = [letter]
                for landmark in hand_landmarks.landmark:
                    row.append(landmark.x)
                    row.append(landmark.y)
                data_to_append.append(row)
                frames_collected += 1
                
    # Draw UI instructions on the camera window
    if recording:
        cv2.putText(frame, f"RECORDING '{letter}': {frames_collected}/{num_frames_to_collect}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if frames_collected >= num_frames_to_collect:
            break # Stop once we hit the target
    else:
        cv2.putText(frame, f"Press 's' to start recording '{letter}'", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    cv2.imshow('Custom Data Collection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        recording = True
        print(f"Recording {num_frames_to_collect} frames...")
    elif key == 27: # ESC key
        print("Cancelled.")
        break

cap.release()
cv2.destroyAllWindows()

# 3. Save the new data into the massive spreadsheet
if frames_collected > 0:
    print(f"\nSaving {frames_collected} frames to {DATA_FILE}...")
    
    # Use pandas to append directly to the bottom of the CSV without overwriting headers
    df_new = pd.DataFrame(data_to_append)
    df_new.to_csv(DATA_FILE, mode='a', header=False, index=False)
    
    print(f"Successfully added your personal hand data for '{letter}'!")
    print("Now run 'python train_upgraded_model.py' to teach the AI your specific hand signs.")
else:
    print("No data collected.")
