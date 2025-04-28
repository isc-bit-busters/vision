# filepath: /home/marta/Projects/vision/camera.py
import cv2
camera_index = 4 # Change this index to test different cameras

cap = cv2.VideoCapture(camera_index)
#agmented camera view
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Set the camera resolution to 1920x1080
cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frame rate to 30 FPS

if cap.isOpened():
    print(f"Camera found at index {camera_index}")
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    h, w = frame.shape[:2]
    if ret:
        image_path = f"captured_image_{camera_index}.jpg"
        cv2.imwrite(image_path, frame)
        print(f"Image saved to {image_path}")
    else:
        print(f"Error capturing image at index {camera_index}")
    cap.release()
    cv2.destroyAllWindows()
else:
    print(f"Camera not found at index {camera_index}")# filepath: /home/marta/Projects/vision/camera.py

print("Scanning for available cameras...")
# for i in range(10):  # Try 0 through 9
#     cap = cv2.VideoCapture(i)
#     if cap.isOpened():
#         print(f"✅ Camera found at index {i}")
#         cap.release()
#     else:
#         print(f"❌ No camera at index {i}")
