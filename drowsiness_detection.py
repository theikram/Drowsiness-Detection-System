import cv2
import dlib
import numpy as np  # Make sure numpy is imported
import time

print("Opening camera...")

# Load the shape predictor (make sure the path is correct)
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def eye_aspect_ratio(eye):
    # Calculate the Eye Aspect Ratio (EAR) to detect blink
    A = cv2.norm(eye[1] - eye[5]) + cv2.norm(eye[2] - eye[4])
    B = 2.0 * cv2.norm(eye[0] - eye[3])
    return A / B

def drowsiness_detection():
    # Define eye aspect ratio threshold and consecutive frames to trigger drowsiness
    EYE_ASPECT_RATIO_THRESHOLD = 0.3
    CONSECUTIVE_FRAMES = 20
    
    blink_count = 0
    consecutive_blinks = 0
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Extract eye landmarks (left and right eyes)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Convert eye landmarks to numpy arrays for easier calculation
            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)

            # Compute the Eye Aspect Ratio (EAR) for both eyes
            ear_left = eye_aspect_ratio(left_eye)
            ear_right = eye_aspect_ratio(right_eye)

            # Average EAR for both eyes
            ear = (ear_left + ear_right) / 2.0

            # Check if EAR is below threshold, indicating a blink
            if ear < EYE_ASPECT_RATIO_THRESHOLD:
                consecutive_blinks += 1
                if consecutive_blinks >= CONSECUTIVE_FRAMES:
                    blink_count += 1
                    consecutive_blinks = 0
                    print(f"Blink detected! Total blinks: {blink_count}")
            else:
                consecutive_blinks = 0

        # Display the frame with facial landmarks and the current blink count
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Drowsiness Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

drowsiness_detection()
