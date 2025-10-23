from flask import Flask, render_template, Response, jsonify, redirect
from flask_socketio import SocketIO
from detector_class import DrowsinessDetector
import logging
import os
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize detector
detector = None

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
    global detector
    if detector is None:
        try:
            detector = DrowsinessDetector(socketio)
        except Exception as e:
            logging.error(f"Failed to initialize detector: {e}")
            return "Failed to initialize camera", 500
    
    return Response(detector.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health_check():
    global detector
    status = {
        'camera': detector is not None and detector.camera.camera.isOpened() if detector else False,
        'face_detector': detector is not None and detector.detector is not None if detector else False,
        'predictor': detector is not None and detector.predictor is not None if detector else False,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/start-detection/<camera_type>', methods=['POST', 'GET'])
def start_detection(camera_type):
    try:
        if camera_type not in ['webcam', 'phone']:
            raise ValueError("Invalid camera type. Allowed types: 'webcam', 'phone'")

        if camera_type == 'webcam':
            return redirect('/video_feed')
        elif camera_type == 'phone':
            script = "android_cam.py"
            if not os.path.isfile(script):
                raise FileNotFoundError(f"Script {script} not found")
            
            process = subprocess.Popen(['python', script, camera_type], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
            return "Detection started successfully"

    except Exception as e:
        logging.error(f"Error starting detection: {e}")
        return str(e), 500

@socketio.on('disconnect')
def handle_disconnect():
    global detector
    if detector:
        detector.cleanup()
        detector = None
    logging.info("Client disconnected, cleaned up resources")

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        socketio.run(app, host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logging.error(f"Application failed to start: {e}")
    finally:
        if detector:
            detector.cleanup()