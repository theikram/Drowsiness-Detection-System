import cv2
import logging
import time

class CameraManager:
    def __init__(self):
        self.camera = None
        self.initialize_camera()

    def initialize_camera(self):
        """Initialize camera with multiple attempts and backends"""
        backends = [
            (cv2.CAP_ANY, "Default"),
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation")
        ]
        
        for camera_index in range(2):  # Try first two camera indices
            for backend, backend_name in backends:
                try:
                    logging.info(f"Trying camera {camera_index} with {backend_name}")
                    self.camera = cv2.VideoCapture(camera_index + backend)
                    
                    if self.camera.isOpened():
                        # Configure camera settings
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.camera.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Test read
                        ret, frame = self.camera.read()
                        if ret and frame is not None:
                            logging.info(f"Successfully initialized camera {camera_index} with {backend_name}")
                            return
                        
                        self.camera.release()
                except Exception as e:
                    logging.warning(f"Failed to initialize camera {camera_index} with {backend_name}: {e}")
                    if self.camera:
                        self.camera.release()
                        
        raise RuntimeError("Failed to initialize camera with any backend")

    def read_frame(self):
        """Read a frame from the camera with error handling and recovery"""
        if not self.camera or not self.camera.isOpened():
            try:
                self.initialize_camera()
            except Exception as e:
                logging.error(f"Failed to reinitialize camera: {e}")
                return None

        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                logging.warning("Failed to read frame, attempting to reinitialize camera")
                self.initialize_camera()
                return None
            return frame
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            return None

    def release(self):
        """Release the camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None