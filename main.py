import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video stream")

first_eye_distance = None
flip_image = False

M_shift = None
M_rotate = None
M_scale = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    
    if len(faces) == 0 and first_eye_distance is not None:
        shifted_frame = cv2.warpAffine(frame, M_shift, (width, height))

        rotated_frame = cv2.warpAffine(shifted_frame, M_rotate, (width, height))

        final_frame = cv2.warpAffine(rotated_frame, M_scale, (width, height))

        frame = final_frame

    for face in faces:
        shape = predictor(gray, face)

        right_eye_center = np.mean(np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]), axis=0).astype(int)
        left_eye_center = np.mean(np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]), axis=0).astype(int)

        center_of_eyes = np.mean(np.array([right_eye_center, left_eye_center]), axis=0).astype(int)

        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)

        if first_eye_distance is None:
            first_eye_distance = eye_distance

        scale_factor = first_eye_distance / eye_distance

        print(f"Right eye center: {right_eye_center}")
        print(f"Left eye center: {left_eye_center}")
        print(f"Center of eyes: {center_of_eyes}")

        for i in range(68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0, 255, 0), thickness=2)

        eye_x, eye_y = center_of_eyes
        shift_x = width // 2 - eye_x
        shift_y = height // 2 - eye_y

        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        shifted_frame = cv2.warpAffine(frame, M_shift, (width, height))

        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x)) - 180

        M_rotate = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        rotated_frame = cv2.warpAffine(shifted_frame, M_rotate, (width, height))

        M_scale = cv2.getRotationMatrix2D((width // 2, height // 2), 0, scale_factor)
        final_frame = cv2.warpAffine(rotated_frame, M_scale, (width, height))

        frame = final_frame

        if flip_image:
            frame = cv2.flip(frame, 1)

    cv2.imshow('Webcam Feed', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key == ord(' '):
        flip_image = not flip_image

cap.release()
cv2.destroyAllWindows()
