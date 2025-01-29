import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define a function to count raised fingers
def count_fingers(hand_landmarks):
    # Finger tip landmarks
    FINGER_TIPS = [ 4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

    # Check if each finger is raised
    fingers = []
    for i, tip in enumerate(FINGER_TIPS):
        if i == 0:  # Special case for the thumb
            fingers.append(
                hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x
            )
        else:  # Other fingers
            fingers.append(
                hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y
            )

    return fingers.count(True)  # Count the number of raised fingers


# Start the webcam feed
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert the frame to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        result = hands.process(rgb_frame)

        # Draw landmarks and count fingers if a hand is detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Count raised fingers
                fingers_up = count_fingers(hand_landmarks)

                # Display the count on the screen
                cv2.putText(
                    frame,
                    f"Fingers: {fingers_up}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        # Show the output
        cv2.imshow("Finger Counter", frame)

      
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
