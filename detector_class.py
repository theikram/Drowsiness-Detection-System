import cv2
import dlib
import numpy as np
import pygame
import logging
import os
from camera_manager import CameraManager

class DrowsinessDetector:
    def __init__(self, socketio):
        self.socketio = socketio
        self.camera = CameraManager()
        self.detector = dlib.get_frontal_face_detector()
        
        # Initialize facial landmark predictor
        model_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Facial landmark predictor model not found")
        self.predictor = dlib.shape_predictor(model_path)
        
        # Initialize sound
        pygame.mixer.init()
        
        # Constants
        self.EYE_AR_THRESH = 0.25
        self.EYE_AR_CONSEC_FRAMES = 20
        self.MOUTH_AR_THRESH = 0.5
        
        # Counters
        self.eye_counter = 0
        self.mouth_counter = 0

    def calculate_ear(self, eye_points):
        # Calculate the vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Calculate the horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_mar(self, mouth_points):
        # Calculate vertical distances
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])
        
        # Calculate horizontal distance
        C = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        # Calculate MAR
        mar = (A + B) / (2.0 * C)
        return mar

    def play_alarm(self, alarm_type):
        sound_file = "alarm.mp3" if alarm_type == "eyes" else "yawn_alarm.mp3"
        try:
            sound_path = os.path.join("sounds", sound_file)
            if os.path.exists(sound_path):
                pygame.mixer.music.load(sound_path)
                pygame.mixer.music.play()
        except Exception as e:
            logging.error(f"Failed to play alarm: {e}")

    def process_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray, 0)
        
        for face in faces:
            # Get facial landmarks
            shape = self.predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])
            
            # Get specific landmarks for eyes and mouth
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            mouth = shape[48:68]
            
            # Calculate EAR and MAR
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = self.calculate_mar(mouth)
            
            # Check for eye closure
            if ear < self.EYE_AR_THRESH:
                self.eye_counter += 1
                if self.eye_counter >= self.EYE_AR_CONSEC_FRAMES:
                    self.socketio.emit('alert', {'type': 'eyes', 'message': 'Drowsiness detected!'})
                    self.play_alarm("eyes")
            else:
                self.eye_counter = 0
            
            # Check for yawning
            if mar > self.MOUTH_AR_THRESH:
                self.mouth_counter += 1
                if self.mouth_counter >= 15:  # Adjust threshold as needed
                    self.socketio.emit('alert', {'type': 'yawn', 'message': 'Yawning detected!'})
                    self.play_alarm("yawn")
            else:
                self.mouth_counter = 0
            
            # Draw landmarks
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        return frame

    def generate_frames(self):
        while True:
            frame = self.camera.read_frame()
            if frame is None:
                continue
                
            # Process frame for drowsiness detection
            processed_frame = self.process_frame(frame)
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def cleanup(self):
        self.camera.release()
        pygame.mixer.quit()