import os
from pathlib import Path
import numpy as np
import pyvista as pv
import pyminiply
import cv2
from scipy.spatial.transform import Rotation as R
from config import RAW_DATA_DIR, SCREENSHOTS_OBJ_DIRS, NORMALS_OBJ_DIRS


def process_orientation(idx, verts, norms, ax, ay, az, ss_dir, n_dir):
    # build rotation matrix (Z-Y-X)
    Rm = (
        R.from_rotvec(np.radians(az) * np.array([0, 0, 1])) *
        R.from_rotvec(np.radians(ay) * np.array([0, 1, 0])) *
        R.from_rotvec(np.radians(ax) * np.array([1, 0, 0]))
    ).as_matrix()
    rv = verts @ Rm.T
    rn = norms @ Rm.T
    rn /= np.linalg.norm(rn, axis=1)[:, None]

    # setup mesh
    mesh = pv.PolyData(rv)
    mesh.point_data["Normals"] = rn
    mesh.point_data["Normals_rgb"] = (rn + 1.0) / 2.0

    def setup_plotter():
        p = pv.Plotter(off_screen=True, window_size=(800, 800))
        p.hide_axes()
        p.set_background("white")
        return p

    # regular screenshot
    p = setup_plotter()
    p.add_mesh(mesh, scalars="Normals", lighting=True, smooth_shading=True, show_scalar_bar=False)
    p.camera_position = 'xy'
    p.enable_anti_aliasing()
    p.render()
    img = p.screenshot(transparent_background=False)
    cam_spec = p.camera_position
    p.close()
    if img is not None:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(Path(ss_dir) / f"screenshot_obj_{idx:02d}_{ax}_{ay}_{az}.png"), bgr)

    # normal-map screenshot
    pn = setup_plotter()
    pn.add_mesh(mesh, scalars="Normals_rgb", rgb=True, lighting=False)
    pn.camera_position = cam_spec
    pn.render()
    nm = pn.screenshot(transparent_background=False)
    pn.close()
    if nm is not None:
        bgr = cv2.cvtColor(nm, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(str(Path(n_dir) / f"normalmap_obj_{idx:02d}_{ax}_{ay}_{az}.png"), bgr)


if __name__ == "__main__":
    # Read models and prepare object folders
    for i in range(1, 15):
        ply_path = Path(RAW_DATA_DIR) / f"obj_{i:02d}.ply"
        if not ply_path.exists():
            print(f"Skipping missing: {ply_path}")
            continue

        ss_dir = SCREENSHOTS_OBJ_DIRS[f"obj_{i:02d}"]
        n_dir = NORMALS_OBJ_DIRS[f"obj_{i:02d}"]
        os.makedirs(ss_dir, exist_ok=True)
        os.makedirs(n_dir, exist_ok=True)

        mesh = pyminiply.read_as_mesh(str(ply_path))
        V = np.array(mesh.points)
        N = np.array(mesh.point_normals)
        if V.size == 0:
            continue
        N /= np.linalg.norm(N, axis=1)[:, None]

        for ax in range(0, 361, 45):
            for ay in range(0, 361, 45):
                for az in range(0, 361, 45):
                    process_orientation(i, V, N, ax, ay, az, ss_dir, n_dir)

    print("Done. Screenshots and normals saved under data/postprocessed_data/")
