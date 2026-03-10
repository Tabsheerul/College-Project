import cv2
import mediapipe as mp

# 1. Initialize Google MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 2. Set up the AI model rules
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,         # We will track just 1 hand to keep it simple
    min_detection_confidence=0.7, # The AI must be 70% sure it's a hand
    min_tracking_confidence=0.5
)

# 3. Turn on the webcam
cap = cv2.VideoCapture(0)
print("Starting up the upgraded AI Camera... Press 'Esc' to close.")

while True:
    success, frame = cap.read()
    if not success:
        print("Camera not working.")
        break

    # Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)

    # MediaPipe needs colors in RGB format, but OpenCV reads in BGR. Let's convert it!
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. Feed the image to the AI to find the hand landmarks
    results = hands.process(rgb_frame)

    # 5. If it finds a hand, draw the skeleton!
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # Draw the dots and lines on your hand
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4), # Green dots
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)  # Blue lines
            )

    # Show the final image to you
    cv2.imshow('Upgraded MediaPipe Hand Tracker', frame)

    # Close if 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up when done
cap.release()
cv2.destroyAllWindows()