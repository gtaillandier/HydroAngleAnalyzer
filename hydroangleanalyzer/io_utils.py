from ovito.io import import_file
import numpy as np
import os

def load_dump_ovito(in_path):
    pipeline = import_file(in_path)
    # Add necessary modifiers
    return pipeline

def save_array_as_txt(array, filename):
    np.savetxt(filename, array, fmt='%f')
def geometric_center(list_xyz_point):
    return np.mean(list_xyz_point, axis=0)


def detect_parser_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.dump', '.lammpstrj']:
        return 'dump'
    elif ext in ['.traj']:
        return 'ase'
    elif ext in ['.xyz']:
        return 'xyz'
    else:
        raise ValueError(f"Unsupported trajectory file format: {ext}")