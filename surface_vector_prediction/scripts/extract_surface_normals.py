import os
import numpy as np
import pyvista as pv
import pyminiply
from scipy.spatial.transform import Rotation as R
import cv2

# Define paths
here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(here, os.pardir))
models_dir = os.path.join(project_root, "data", "raw")
augment_dir = os.path.join(project_root, "data", "data_augmentation")
os.makedirs(augment_dir, exist_ok=True)

model_file = os.path.join(models_dir, "obj_15.ply")

# Load mesh using pyminiply
mesh = pyminiply.read_as_mesh(model_file)

# Extract vertices and normals
vertices = np.array(mesh.points)
normals = np.array(mesh.point_normals)

# Save original data
original_data = np.hstack((vertices, normals))
np.save(os.path.join(augment_dir, "original_obj_15.npy"), original_data)


def rotate_and_save(axis_name, angle_deg):
    # Create quaternion rotation about a single axis
    axis_map = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
    axis = axis_map[axis_name]
    rotation = R.from_rotvec(np.radians(angle_deg) * np.array(axis))
    rot_matrix = rotation.as_matrix()

    # Rotate vertices and normals
    rotated_vertices = vertices @ rot_matrix.T
    rotated_normals = normals @ rot_matrix.T

    # Create mesh
    mesh_pv = pv.PolyData(rotated_vertices)
    mesh_pv["Normals"] = rotated_normals

    # Get visible cells (for visible normals)
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.add_mesh(mesh_pv, scalars="Normals", lighting=True)
    plotter.camera_position = 'xy'
    plotter.show(auto_close=False)

    camera_pos = np.array(plotter.camera.GetPosition())
    visible_mask = filter_visible_normals(rotated_normals, rotated_vertices, camera_pos)
    visible_normals = rotated_normals[visible_mask]


    # Screenshot
    image = plotter.screenshot(transparent_background=True)
    cv2_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # Save screenshot
    screenshot_filename = f"screenshot_rot_{axis_name}_{angle_deg}deg.png"
    screenshot_path = os.path.join(augment_dir, screenshot_filename)
    cv2.imwrite(screenshot_path, cv2_image)

    # Save visible normals
    normals_filename = f"visible_normals_rot_{axis_name}_{angle_deg}deg.npy"
    normals_path = os.path.join(augment_dir, normals_filename)
    np.save(normals_path, visible_normals)

    plotter.close()

def filter_visible_normals(normals, vertices, camera_pos):
    # Compute view direction vector from camera to each vertex
    view_vectors = vertices - camera_pos
    view_vectors /= np.linalg.norm(view_vectors, axis=1, keepdims=True)

    # Compute dot product of normal and view vector
    dot_products = np.sum(normals * view_vectors, axis=1)

    # Normals facing the camera will have negative dot product
    visible_mask = dot_products < 0
    return visible_mask


# Rotation for each axis in steps of 90 degrees as an example
for axis in ['x', 'y', 'z']:
    for angle in range(0, 361, 90):
        rotate_and_save(axis, angle)


print("All rotations, screenshots, and data saving completed successfully.")