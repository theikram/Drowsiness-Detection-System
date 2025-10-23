import requests
import numpy as np
from imutils import face_utils 
import imutils 
import dlib
import time 
import argparse 
import cv2 
import os 
from datetime import datetime
from pygame import mixer  # Using pygame instead of playsound for better performance

# Initialize pygame mixer for sound
mixer.init()
alarm_sound = mixer.Sound('soundfiles/warning.mp3')
warning_sound = mixer.Sound('soundfiles/alarm.mp3')
warning_yawn_sound = mixer.Sound('soundfiles/warning_yawn.mp3')

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Compute the euclidean distances between the vertical mouth landmarks
    A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[8])   # 53, 57
    # Compute the euclidean distance between the horizontal mouth landmarks
    C = np.linalg.norm(mouth[0] - mouth[6])   # 49, 55
    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar

# Create dataset directory if it doesn't exist
os.makedirs("dataset_phonecam", exist_ok=True)

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

# Constants for detection
EAR_THRESHOLD = 0.3
CONSECUTIVE_FRAMES = 15
MAR_THRESHOLD = 0.7  # Adjusted MAR threshold
YAWN_CONSECUTIVE_FRAMES = 5  # Added consecutive frames for yawn detection

# Initialize counters
FRAME_COUNT = 0
YAWN_FRAME_COUNT = 0  # Added yawn frame counter
count_sleep = 0
count_yawn = 0

# Initialize face detector and predictor
print("[INFO] Loading the predictor.....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Get facial landmark indexes
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# URL for IP webcam
url = "http://192.168.18.106:8080/shot.jpg"

print("[INFO] Starting video stream...")
time.sleep(1)  # Reduced sleep time for faster startup

last_alert_time = time.time()  # For managing alert frequency

try:
    while True:
        # Get frame from IP webcam
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        frame = cv2.imdecode(img_arr, -1)
        
        # Process frame
        frame = imutils.resize(frame, width=875)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        rects = detector(gray, 0)  # Changed from 1 to 0 for faster detection
        
        # Process each face
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Draw face rectangle
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Get eyes and mouth regions
            leftEye = shape[lstart:lend]
            rightEye = shape[rstart:rend]
            mouth = shape[mstart:mend]
            
            # Calculate EAR and MAR
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            EAR = (leftEAR + rightEAR) / 2.0
            MAR = mouth_aspect_ratio(mouth)
            
            # Draw eye and mouth contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
            
            # Check for eye closure
            if EAR < EAR_THRESHOLD:
                FRAME_COUNT += 1
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
                
                if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    if current_time - last_alert_time > 2.0:  # Minimum 2 seconds between alerts
                        count_sleep += 1
                        cv2.imwrite(f"dataset_phonecam/frame_sleep{count_sleep}.jpg", frame)
                        alarm_sound.play()
                        cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        last_alert_time = current_time
            else:
                FRAME_COUNT = 0
            
            # Check for yawning
            if MAR > MAR_THRESHOLD:
                YAWN_FRAME_COUNT += 1
                cv2.drawContours(frame, [mouth], -1, (255, 0, 0), 1)  # Blue contour for yawn detection
                
                if YAWN_FRAME_COUNT >= YAWN_CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    if current_time - last_alert_time > 2.0:
                        count_yawn += 1
                        cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
                        cv2.putText(frame, "YAWN DETECTED!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imwrite(f"dataset_phonecam/frame_yawn{count_yawn}.jpg", frame)
                        warning_yawn_sound.play()
                        last_alert_time = current_time
            else:
                YAWN_FRAME_COUNT = 0  # Reset yawn counter when mouth closes
        
        cv2.putText(frame, "Press 'q' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()