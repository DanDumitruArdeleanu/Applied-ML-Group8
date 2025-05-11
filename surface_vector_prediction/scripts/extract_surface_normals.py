import os
import numpy as np
import pyvista as pv
import pyminiply
from scipy.spatial.transform import Rotation as R

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

# Define rotation angles (example: rotate 45 degrees around Y-axis)
rotation_angles = (0, 45, 0)  # (x, y, z) in degrees

# Create rotation matrix
rotation = R.from_euler('xyz', rotation_angles, degrees=True)
rot_matrix = rotation.as_matrix()

# Rotate vertices and normals
rotated_vertices = vertices @ rot_matrix.T
rotated_normals = normals @ rot_matrix.T

# Save rotated data
rotated_data = np.hstack((rotated_vertices, rotated_normals))
np.save(os.path.join(augment_dir, "rotated_obj_15_y45.npy"), rotated_data)

# Visualize and screenshot using pyvista
plotter = pv.Plotter(off_screen=True)
mesh_pv = pv.PolyData(rotated_vertices)
mesh_pv["Normals"] = rotated_normals
plotter.add_mesh(mesh_pv, scalars="Normals", lighting=True)
screenshot_path = os.path.join(augment_dir, "rotated_obj_15_y45.png")
plotter.show(screenshot=screenshot_path)

plotter.close()

print("Rotation, screenshot, and data save completed successfully.")
