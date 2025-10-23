from flask_socketio import SocketIO, emit
import cv2
import dlib
import numpy as np
import subprocess
import os
import pygame
import time
from flask import Flask, render_template, request, jsonify, Response, redirect
import logging
import atexit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask app and socket
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize pygame mixer with specific settings
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    logger.info("Pygame mixer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pygame mixer: {e}")
    raise

# Load sound files once at startup
try:
    warning_sound = pygame.mixer.Sound('soundfiles/warning.mp3')
    yawn_sound = pygame.mixer.Sound('soundfiles/warning_yawn.mp3')
    logger.info("Sound files loaded successfully")
except Exception as e:
    logger.error(f"Error loading sound files: {e}")
    raise

# Initialize the webcam
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Failed to open webcam")
    logger.info("Webcam initialized successfully")
except Exception as e:
    logger.error(f"Error initializing webcam: {e}")
    raise

# Initialize face detection
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    logger.info("Face detection initialized successfully")
except Exception as e:
    logger.error(f"Error initializing face detection: {e}")
    raise

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the vertical mouth landmarks
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    # Compute the euclidean distance between the horizontal mouth landmarks
    C = np.linalg.norm(mouth[0] - mouth[6])
    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

def play_sound(sound_type):
    try:
        if sound_type == "warning":
            if not pygame.mixer.get_busy():
                warning_sound.play()
                logger.info("Playing warning sound")
        elif sound_type == "yawn":
            if not pygame.mixer.get_busy():
                yawn_sound.play()
                logger.info("Playing yawn sound")
    except Exception as e:
        logger.error(f"Error playing sound {sound_type}: {e}")

def calibrate_thresholds(dataset_path):
    try:
        ear_values = []
        mar_values = []
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path {dataset_path} not found. Using default thresholds.")
            return 0.25, 0.6  # Default thresholds
        
        for image_name in os.listdir(dataset_path):
            image_path = os.path.join(dataset_path, image_name)
            if image_path.lower().endswith(('jpg', 'jpeg', 'png')):
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        if not ear_values or not mar_values:
            logger.warning("No valid calibration data found. Using default thresholds.")
            return 0.25, 0.6

        average_ear = np.mean(ear_values)
        average_mar = np.mean(mar_values)
        
        EAR_THRESHOLD = average_ear * 0.85
        MAR_THRESHOLD = average_mar * 1.1
        
        logger.info(f"Calibrated thresholds - EAR: {EAR_THRESHOLD:.3f}, MAR: {MAR_THRESHOLD:.3f}")
        return EAR_THRESHOLD, MAR_THRESHOLD
        
    except Exception as e:
        logger.error(f"Error during calibration: {e}")
        return 0.25, 0.6  # Default thresholds

# Calibrate thresholds
EAR_THRESHOLD, MAR_THRESHOLD = calibrate_thresholds("dataset/")

def detect_drowsiness():
    EYE_CONSEC_FRAMES = 8
    YAWN_CONSEC_FRAMES = 5
    eye_counter = 0
    yawn_counter = 0
    last_alert_time = time.time()
    alert_cooldown = 2.0  # Minimum time between alerts

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to grab frame")
                break

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                landmarks = predictor(gray, face)
                
                # Get eye landmarks and calculate EAR
                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                
                left_eye_ear = eye_aspect_ratio(left_eye)
                right_eye_ear = eye_aspect_ratio(right_eye)
                ear = (left_eye_ear + right_eye_ear) / 2.0
                
                # Display EAR value
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check for drowsiness
                if ear < EAR_THRESHOLD:
                    eye_counter += 1
                    if eye_counter >= EYE_CONSEC_FRAMES:
                        current_time = time.time()
                        if current_time - last_alert_time > alert_cooldown:
                            logger.info(f"Drowsiness detected with EAR: {ear}")
                            play_sound("warning")
                            socketio.emit('drowsiness_alert', {'alert': 'Drowsiness Detected: Eyes Closed'})
                            last_alert_time = current_time
                else:
                    eye_counter = 0
                
                # Get mouth landmarks and calculate MAR
                mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
                mar = mouth_aspect_ratio(mouth)
                
                # Display MAR value
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Check for yawning
                if mar > MAR_THRESHOLD:
                    yawn_counter += 1
                    if yawn_counter >= YAWN_CONSEC_FRAMES:
                        current_time = time.time()
                        if current_time - last_alert_time > alert_cooldown:
                            logger.info(f"Yawn detected with MAR: {mar}")
                            play_sound("yawn")
                            socketio.emit('drowsiness_alert', {'alert': 'Drowsiness Detected: Yawning'})
                            last_alert_time = current_time
                else:
                    yawn_counter = 0
                
                # Draw landmarks
                for (x, y) in left_eye:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
                for (x, y) in right_eye:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
                for (x, y) in mouth:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            # Display the frame
            cv2.imshow("Drowsiness Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Encode frame for web streaming
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    except Exception as e:
        logger.error(f"Error in detection loop: {e}")
    
    finally:
        logger.info("Stopping detection")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/understanding')
def understanding():
    return render_template('understanding.html')

@app.route('/levels')
def levels():
    return render_template('levels.html')

@app.route('/how')
def how():
    return render_template('how.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-detection/<camera_type>', methods=['POST', 'GET'])
def start_detection(camera_type):
    try:
        if camera_type not in ['webcam', 'phone']:
            raise ValueError("Invalid camera type")
            
        if camera_type == 'webcam':
            return redirect('/video_feed')
        elif camera_type == 'phone':
            process = subprocess.Popen(
                ["python", "android_cam.py", "--shape_predictor", "shape_predictor_68_face_landmarks.dat"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return jsonify({'status': 'Detection started with phone camera'})
            
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        return jsonify({'error': str(e)}), 500

# Cleanup function
@atexit.register
def cleanup():
    try:
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
    finally:
        cleanup()