import os
import pyminiply

here = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(here, os.pardir))
models_dir   = os.path.join(project_root, "models")
model_file   = os.path.join(models_dir, "obj_01.ply")


mesh = pyminiply.read_as_mesh(model_file)
mesh.plot()
