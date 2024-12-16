import cv2
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera. Make sure it is connected and enabled.")
    exit()

print("Press 'q' to exit.")

# Function to determine if a finger is folded
def is_finger_folded(landmarks, tip, pip, dip):
    """Returns True if the finger is folded, False if extended"""
    # Check the angle between the tip, pip (proximal interphalangeal), and dip (distal interphalangeal)
    # If the angle is large, the finger is folded, otherwise, it's extended
    angle = (pip.y - tip.y) * (dip.y - pip.y)  # Angle-like comparison between segments
    return angle > 0.03  # A threshold to decide if folded (adjust as necessary)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera. Exiting...")
        break

    # Flip the frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (Mediapipe expects RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Define landmarks for each finger (thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20)
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            finger_landmarks = [4, 8, 12, 16, 20]  # Landmarks for each finger's tip

            # Initialize dictionary to store finger statuses
            finger_status = {}

            # Check each finger's fold status by comparing its joints
            for i, finger in enumerate(finger_names):
                tip = hand_landmarks.landmark[finger_landmarks[i]]
                pip = hand_landmarks.landmark[finger_landmarks[i] - 2]  # Proximal joint
                dip = hand_landmarks.landmark[finger_landmarks[i] - 1]  # Distal joint

                # Determine if the finger is folded
                if is_finger_folded(hand_landmarks.landmark, tip, pip, dip):
                    finger_status[finger] = "Folded"
                else:
                    finger_status[finger] = "Extended"

                # Display finger status on the frame
                cv2.putText(frame, f"{finger}: {finger_status[finger]}", (50, 50 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with overlays
    cv2.imshow("Finger Counter", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
