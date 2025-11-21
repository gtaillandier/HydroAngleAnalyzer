import os

import numpy as np


def load_dump_ovito(in_path):
    try:
        from ovito.io import import_file
    except ImportError as e:  # add exception chaining
        raise ImportError(
            "The 'ovito' package is required for load dump_ovito. Install it with: "
            "pip install HydroAngleAnalyzer[ovito]"
        ) from e
    pipeline = import_file(in_path)
    # Add necessary modifiers
    return pipeline


def save_array_as_txt(array, filename):
    np.savetxt(filename, array, fmt="%f")


def geometric_center(list_xyz_point):
    return np.mean(list_xyz_point, axis=0)


def detect_parser_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".dump", ".lammpstrj"]:
        return "dump"
    if ext in [".traj"]:
        return "ase"
    if ext in [".xyz"]:
        return "xyz"
    raise ValueError(f"Unsupported trajectory file format: {ext}")
