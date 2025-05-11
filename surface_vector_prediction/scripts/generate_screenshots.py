import os
import pyminiply
import pyvista as pv
import numpy as np
from typing import List


def ensure_dir(directory: str) -> str:
    """Ensure that the given directory exists."""
    os.makedirs(directory, exist_ok=True)
    return directory


def generate_spherical_camera_positions(n: int, radius: float) -> List[List[float]]:
    """Generate n evenly distributed points on a sphere using the Fibonacci lattice."""
    points = []
    offset = 2.0 / n
    increment = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(1 - y * y)
        phi = i * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        points.append([radius * x, radius * y, radius * z])
    return points


def generate_screenshots(models_dir: str, screenshots_dir: str, n_views: int = 15) -> None:
    """Generate screenshots for each model in the models directory."""
    for i in range(1, 3):
        model_filename = f"obj_{i:02d}.ply"
        model_path = os.path.join(models_dir, model_filename)

        if os.path.exists(model_path):
            mesh = pyminiply.read_as_mesh(model_path)
            points = np.array(mesh.points)
            
            # Ensure faces are correctly processed
            faces = mesh.faces
            if isinstance(faces, list):  # If faces are already a list of lists, flatten
                faces_flat = np.hstack([[len(f), *f] for f in faces]).astype(np.int32)
            else:
                # If faces are just a flat array of integers, reshape and organize them
                faces_flat = np.array(faces).flatten()

            # Ensure the number of face elements is divisible by 4 (since each face should have 4 numbers: [count, v1, v2, v3])
            if len(faces_flat) % 4 != 0:
                raise ValueError(f"Faces array is not divisible by 4. Length: {len(faces_flat)}")

            # Correctly format the faces array as [count, v1, v2, v3]
            faces_reshaped = faces_flat.reshape((-1, 4))

            # Create the PyVista mesh with correctly formatted faces
            pyvista_mesh = pv.PolyData(points)
            pyvista_mesh.faces = faces_reshaped.flatten()

            # Add colors if they exist
            if hasattr(mesh, "red") and hasattr(mesh, "green") and hasattr(mesh, "blue"):
                colors = np.stack([mesh.red, mesh.green, mesh.blue], axis=1)
                pyvista_mesh.point_data["RGB"] = colors

            bounds = pyvista_mesh.bounds
            center = pyvista_mesh.center
            max_extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            radius = max_extent * 2.5

            camera_positions = generate_spherical_camera_positions(n_views, radius)

            model_screenshot_dir = os.path.join(screenshots_dir, f"obj_{i:02d}")
            ensure_dir(model_screenshot_dir)

            for j, pos in enumerate(camera_positions):
                plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
                plotter.add_mesh(
                    pyvista_mesh,
                    scalars="RGB" if "RGB" in pyvista_mesh.point_data else None,
                    rgb="RGB" in pyvista_mesh.point_data,
                    smooth_shading=True,
                    specular=0.5,
                    specular_power=15,
                    lighting=True,
                )
                plotter.camera_position = [pos, center, [0, 0, 1]]
                plotter.set_background("white")
                screenshot_path = os.path.join(model_screenshot_dir, f"view{j + 1}.png")
                plotter.show(screenshot=screenshot_path)
                plotter.close()


def main() -> None:
    """Main entry point to generate screenshots for all models."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    models_dir = os.path.join(project_root, "models")
    screenshots_dir = os.path.join(project_root, "data", "screenshots_test")

    ensure_dir(screenshots_dir)
    generate_screenshots(models_dir, screenshots_dir, n_views=2)


if __name__ == "__main__":
    main()
