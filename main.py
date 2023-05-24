import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start the webcam with OpenCV
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video stream")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Get the size of the frame
    height, width = frame.shape[:2]

    # Convert color to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Iterate over the faces detected
    for face in faces:
        # Get the landmarks/parts for the face
        shape = predictor(gray, face)

        # Get coordinates for the centers of the eyes
        right_eye_center = np.mean(np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]), axis=0).astype(int)
        left_eye_center = np.mean(np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]), axis=0).astype(int)

        # Calculate the center point between the two eyes
        center_of_eyes = np.mean(np.array([right_eye_center, left_eye_center]), axis=0).astype(int)

        # Print the coordinates
        print(f"Right eye center: {right_eye_center}")
        print(f"Left eye center: {left_eye_center}")
        print(f"Center of eyes: {center_of_eyes}")

        # Draw small circles at each of the 68 facial landmarks
        for i in range(68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), thickness=2)

        # Shift frame around the center of eyes
        eye_x, eye_y = center_of_eyes
        shift_x = width // 2 - eye_x
        shift_y = height // 2 - eye_y

        # Define transformation matrix for shift
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_frame = cv2.warpAffine(frame, M_shift, (width, height))

        # Compute the angle between the line connecting the centers of the eyes and the horizontal line
        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Rotate frame to make the line between the eyes horizontal
        M_rotate = cv2.getRotationMatrix2D((width // 2, height // 2), angle - 180, 1)
        rotated_frame = cv2.warpAffine(shifted_frame, M_rotate, (width, height))

        frame = rotated_frame

    # Display the resulting frame with landmarks
    cv2.imshow('Webcam Feed', frame)
    # Break the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
