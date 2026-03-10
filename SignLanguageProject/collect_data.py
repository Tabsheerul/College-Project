import cv2
import mediapipe as mp
import csv
import os

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 2. Ask what gesture we are recording
gesture_name = input("Enter the name of the gesture you want to record (e.g., A, B, C): ")
print(f"Get ready to record '{gesture_name}'!")
print("Press 'r' on your keyboard when your hand is ready to start recording.")

# 3. Setup the Spreadsheet (CSV file)
csv_file = "gesture_dataset.csv"

# If the file doesn't exist yet, we create it and add the header row
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # We need an X and Y coordinate for all 21 joints (42 columns total), plus the label
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}'])
        writer.writerow(header)

# 4. Start Camera
cap = cv2.VideoCapture(0)
recording = False
frames_recorded = 0
max_frames = 200  # We will capture 200 frames of math data

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 5. Look for the hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 6. If recording is active, grab the math!
            if recording:
                row_data = [gesture_name] # First column is the letter (e.g., 'A')
                for landmark in hand_landmarks.landmark:
                    row_data.append(landmark.x)
                    row_data.append(landmark.y)
                
                # Save it to the spreadsheet
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                
                frames_recorded += 1
                cv2.putText(frame, f"Recording: {frames_recorded}/{max_frames}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if frames_recorded >= max_frames:
                    print(f"Successfully recorded 200 frames for '{gesture_name}'!")
                    recording = False # Stop recording

    # Instructions on the screen
    if not recording and frames_recorded < max_frames:
        cv2.putText(frame, "Press 'r' to Start Recording", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif frames_recorded >= max_frames:
        cv2.putText(frame, "Finished! Press 'Esc' to exit.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Upgraded Data Collector', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and not recording and frames_recorded < max_frames:
        print("Recording Started! Please move your hand slightly to capture different angles.")
        recording = True
    elif key == 27: # Esc key
        break

cap.release()
cv2.destroyAllWindows()