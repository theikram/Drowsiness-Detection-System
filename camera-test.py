import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)

def test_camera_access():
    """Test camera access with different backends and indices"""
    print("Testing camera access...")
    
    backends = [
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation")
    ]
    
    working_configs = []
    
    for index in range(2):  # Test first two camera indices
        for backend, backend_name in backends:
            print(f"\nTrying camera {index} with {backend_name} backend...")
            try:
                if backend == cv2.CAP_ANY:
                    cap = cv2.VideoCapture(index)
                else:
                    cap = cv2.VideoCapture(index, backend)
                
                if cap.isOpened():
                    print(f"Successfully opened camera {index} with {backend_name}")
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Successfully read frame from camera {index}")
                        print(f"Frame shape: {frame.shape}")
                        working_configs.append((index, backend_name))
                    else:
                        print(f"Could not read frame from camera {index}")
                    
                    # Get camera properties
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"Camera properties:")
                    print(f"Resolution: {width}x{height}")
                    print(f"FPS: {fps}")
                    
                    cap.release()
                else:
                    print(f"Could not open camera {index} with {backend_name}")
            
            except Exception as e:
                print(f"Error testing camera {index} with {backend_name}: {str(e)}")
            
            finally:
                if 'cap' in locals():
                    cap.release()
                
            time.sleep(1)  # Wait between attempts
    
    return working_configs

if __name__ == "__main__":
    print("Camera Test Utility")
    print("==================")
    
    working_configs = test_camera_access()
    
    print("\nTest Results:")
    print("============")
    if working_configs:
        print("Working camera configurations:")
        for index, backend in working_configs:
            print(f"- Camera {index} with {backend}")
    else:
        print("No working camera configurations found!")
        print("\nPossible solutions:")
        print("1. Check if camera is properly connected")
        print("2. Check if camera is being used by another application")
        print("3. Check camera permissions in your system settings")
        print("4. Try updating your camera drivers")
        print("5. Try restarting your computer")