import cv2

# Open a connection to the external camera (usually index 1 or higher)
camera_index = 1  # Change this index if needed
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Capture a single frame
ret, frame = cap.read()

if ret:
    # Save the captured image to a file
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)
    print(f"Image saved to {image_path}")
else:
    print("Error: Could not capture an image.")

# Release the camera
cap.release()
cv2.destroyAllWindows()