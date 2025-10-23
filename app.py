from flask_socketio import SocketIO, emit
import cv2
import dlib
import numpy as np
import subprocess
import os
import pygame  # For playing sounds
from flask import Flask, render_template, request, jsonify, Response, redirect
from threading import Thread
import subprocess

# Initialize the Flask app and socket
app = Flask(__name__)
socketio = SocketIO(app)

# Dlib's face detector and shape predictor for facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file exists

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize pygame for sound
pygame.mixer.init()

# Function to calculate Eye Aspect Ratio (EAR) for detecting eye closure
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect yawning (Mouth Aspect Ratio)
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Function to play sound based on detection (use your sound files)
def play_sound(file_name):
    sound_path = os.path.join('soundfiles', file_name)
    if os.path.exists(sound_path):
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play()

# Load dataset to calibrate EAR and MAR thresholds
def calibrate_thresholds(dataset_path):
    ear_values = []
    mar_values = []
    
    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        if image_path.lower().endswith(('jpg', 'jpeg', 'png')):  # Process image files only
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
                
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
                mar = mouth_aspect_ratio(mouth)

                ear_values.append(ear)
                mar_values.append(mar)

    average_ear = np.mean(ear_values)
    average_mar = np.mean(mar_values)

    # Adjust thresholds based on dataset
    EAR_THRESHOLD = average_ear * 0.25  # Adjust as necessary
    MAR_THRESHOLD = average_mar * 0.6   # Adjust as necessary

    return EAR_THRESHOLD, MAR_THRESHOLD

# Calibrate the thresholds
EAR_THRESHOLD, MAR_THRESHOLD = calibrate_thresholds("dataset/")

# Function to detect drowsiness and yawning
def detect_drowsiness():
    EYE_CONSEC_FRAMES = 8
    YAWN_CONSEC_FRAMES = 5

    eye_counter = 0
    yawn_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for better performance
        frame = cv2.resize(frame, (640, 480))  # Resize for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)

            # Get eye landmarks for both eyes
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            # Calculate EAR for both eyes
            left_eye_ear = eye_aspect_ratio(left_eye)
            right_eye_ear = eye_aspect_ratio(right_eye)

            # Check if eyes are closed
            if left_eye_ear < EAR_THRESHOLD and right_eye_ear < EAR_THRESHOLD:
                eye_counter += 1
                if eye_counter >= EYE_CONSEC_FRAMES:
                    alert = "Drowsiness Detected: Eyes Closed"
                    print("sleep detected")
                    socketio.emit('drowsiness_alert', {'alert': alert})
                    play_sound("warning.mp3")  # Play sleep sound
            else:
                eye_counter = 0

            # Get mouth landmarks for yawning detection
            mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
            mar = mouth_aspect_ratio(mouth)

            # Check if yawning
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    alert = "Drowsiness Detected: Yawning"
                    print("Dont Yawn")
                    socketio.emit('drowsiness_alert', {'alert': alert})
                    play_sound("warning_yawn.mp3")  # Play yawning sound
            else:
                yawn_counter = 0

            # Draw the face landmarks for visualization
            for (x, y) in left_eye:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            for (x, y) in mouth:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Show the frame locally (on your laptop screen)
        cv2.imshow("Webcam Feed", frame)

        # Encode the frame to send to the client
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        # Yield the frame for streaming via Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the detection page
@app.route('/detect')
def detect():
    return render_template('detect.html')

# Route for the understanding page
@app.route('/understanding')
def understanding():
    return render_template('understanding.html')

# Route for the levels page
@app.route('/levels')
def levels():
    return render_template('levels.html')

# Route for the "how it works" page
@app.route('/how')
def how():
    return render_template('how.html')

# Route to stream the webcam feed to the browser
@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for starting the detection process with a specific camera type (webcam or phone)
@app.route('/start-detection/<camera_type>', methods=['POST', 'GET'])
def start_detection(camera_type):
    try:
        # Check for valid camera type
        if camera_type not in ['webcam', 'phone']:
            raise ValueError("Invalid camera type. Allowed types: 'webcam', 'phone'")

        # Define script and arguments based on the camera type
        if camera_type == 'webcam':
            return redirect('/video_feed')
        elif camera_type == 'phone':
            subprocess.Popen(["python", "android_cam.py", "--shape_predictor", "shape_predictor_68_face_landmarks.dat"])

        # Ensure the script exists
        if not os.path.isfile(script):
            raise FileNotFoundError(f"The script {script} does not exist or is not accessible.")

        # Execute the detection script with the appropriate arguments
        process = subprocess.Popen(['python', script, camera_type], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return "Detection started successfully."

    except Exception as e:
        return str(e)

# Run Flask app with SocketIO
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)