import cv2
import dlib
import numpy as np

# Load face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define standard eye distance when face is in the center and at a nominal distance from webcam
std_eye_dist = 100.0  # Increase this to make the box appear smaller

# Define camera parameters
focal_length = 50
camera_center = (320, 240)  # Assuming a 640x480 camera

# Define box vertices and edges
box_vertices = np.float32([
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5]
])
box_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Sides
]

# For webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    box_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
    for face in faces:
        landmarks = predictor(gray, face)

        # Calculate the center of each eye
        left_eye_x = int((landmarks.part(36).x + landmarks.part(39).x) / 2)
        left_eye_y = int((landmarks.part(36).y + landmarks.part(39).y) / 2)
        right_eye_x = int((landmarks.part(42).x + landmarks.part(45).x) / 2)
        right_eye_y = int((landmarks.part(42).y + landmarks.part(45).y) / 2)

        # Estimate eye position
        eye_center = [(left_eye_x + right_eye_x) / 2, (left_eye_y + right_eye_y) / 2]
        eye_dist = np.sqrt((left_eye_x - right_eye_x) ** 2 + (left_eye_y - right_eye_y) ** 2)

        x = (eye_center[0] - frame.shape[1] / 2) / frame.shape[1]
        y = (eye_center[1] - frame.shape[0] / 2) / frame.shape[0]
        z = -(1 - (eye_dist / std_eye_dist))

        # Create a blank frame for drawing the box
        box_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Project 3D box vertices to 2D
        eye_position = np.array([x, y, z])
        projected_vertices = []
        for v in box_vertices:
            v -= eye_position
            zz = v[2]
            if zz == 0:
                zz = 0.1  # Avoid division by zero
            projected_x = camera_center[0] + v[0] * focal_length / zz
            projected_y = camera_center[1] + v[1] * focal_length / zz
            projected_vertices.append((int(projected_x), int(projected_y)))

        # Draw edges on the box frame
        for edge in box_edges:
            cv2.line(box_frame, projected_vertices[edge[0]], projected_vertices[edge[1]], (0, 0, 255), 2)
        print(x, y, z)

    # Show the box frame
    cv2.imshow('frame', box_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
