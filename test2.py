import cv2
import numpy as np

# Define camera parameters
focal_length = 500
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

# Create a blank image
image = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Project 3D box vertices to 2D
eye_position = np.array([0, 0, -10])
projected_vertices = []
for v in box_vertices:
    v -= eye_position
    z = v[2]
    if z == 0:
        z = 0.1  # Avoid division by zero
    projected_x = camera_center[0] + v[0] * focal_length / z
    projected_y = camera_center[1] + v[1] * focal_length / z
    projected_vertices.append((int(projected_x), int(projected_y)))

# Draw edges on the image
for edge in box_edges:
    cv2.line(image, projected_vertices[edge[0]], projected_vertices[edge[1]], (0, 0, 0), 1)

# Show the image
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
