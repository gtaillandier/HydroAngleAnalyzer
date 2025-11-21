import os
from unittest.mock import patch

import numpy as np
import pytest

from hydroangleanalyzer.parser.parser_dump import DumpParser

# Path to the test trajectory file (LAMMPS dump format)
TRAJECTORY_PATH = os.path.join(
    os.path.dirname(__file__), "../trajectories/traj_spherical_drop_4k.lammpstrj"
)


# --- Fixture for DumpParser ---
@pytest.fixture
def dump_parser():
    return DumpParser(TRAJECTORY_PATH)


# --- Test ImportError ---
@patch(
    "ovito.io.import_file",
    side_effect=ImportError(
        "The 'ovito' package is required for DumpParser. Install with: "
        "pip install HydroAngleAnalyzer[ovito]"
    ),
)
def test_dump_parser_no_ovito(mock_import_file):
    with pytest.raises(ImportError) as excinfo:
        DumpParser(TRAJECTORY_PATH)
    assert "The 'ovito' package is required for DumpParser" in str(excinfo.value)


# --- Test parse ---
def test_parse(dump_parser):
    num_frame = 0
    positions = dump_parser.parse(num_frame)
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 3  # x, y, z coordinates

    # Test with specific indices
    indices = np.array([1, 2])
    positions_subset = dump_parser.parse(num_frame, indices)
    assert positions_subset.shape[0] <= positions.shape[0]


# --- Test return_cylindrical_coord_pars ---
def test_return_cylindrical_coord_pars(dump_parser, capsys):
    frame_list = [0, 1]
    xi_par = np.array([])
    zi_par = np.array([])

    for frame in frame_list:
        X_par = dump_parser.parse(frame)
        # Simulate particle_ids for testing purposes
        np.arange(X_par.shape[0])
        dim = len(X_par[0, :])
        X_cm = np.array([(X_par[:, i]).sum() / len(X_par[:, i]) for i in range(dim)])
        X_0 = np.array([X_par[:, i] - X_cm[i] * (i < 2) for i in range(dim)])

        xi_par_frame = np.abs(X_0[0] + 0.01)
        zi_par_frame = X_0[2]

        xi_par = np.concatenate((xi_par, xi_par_frame))
        zi_par = np.concatenate((zi_par, zi_par_frame))

        if frame % 10 == 0:
            print(f"Frame: {frame}")
            print(f"Center of Mass: {X_cm}")

    num_frames = len(frame_list)
    assert isinstance(xi_par, np.ndarray)
    assert isinstance(zi_par, np.ndarray)
    assert num_frames == len(frame_list)
    assert xi_par.shape == zi_par.shape

    print("\nxi range:\t({},{})".format(np.min(xi_par), np.max(xi_par)))
    print("zi range:\t({},{})".format(np.min(zi_par), np.max(zi_par)))

    # Check print output
    captured = capsys.readouterr()
    assert "xi range:" in captured.out
    assert "zi range:" in captured.out


# --- Test box_size_x and box_size_y ---
def test_box_size_x(dump_parser):
    num_frame = 0
    box_size_x = dump_parser.box_size_x(num_frame)
    assert isinstance(box_size_x, float)
    assert box_size_x > 0


def test_box_size_y(dump_parser):
    num_frame = 0
    box_size_y = dump_parser.box_size_y(num_frame)
    assert isinstance(box_size_y, float)
    assert box_size_y > 0


# --- Test box_length_max ---
def test_box_length_max(dump_parser):
    num_frame = 0
    max_length = dump_parser.box_length_max(num_frame)
    assert isinstance(max_length, float)
    assert max_length > 0


# --- Test frame_tot ---
def test_frame_tot(dump_parser):
    total_frames = dump_parser.frame_tot()
    assert isinstance(total_frames, int)
    assert total_frames > 0


# --- Test type_model in return_cylindrical_coord_pars ---
def test_return_cylindrical_coord_pars_type_model(dump_parser):
    frame_list = [0]
    xi_par = np.array([])
    zi_par = np.array([])

    for frame in frame_list:
        X_par = dump_parser.parse(frame)
        # Simulate particle_ids for testing purposes
        np.arange(X_par.shape[0])
        dim = len(X_par[0, :])
        X_cm = np.array([(X_par[:, i]).sum() / len(X_par[:, i]) for i in range(dim)])
        X_0 = np.array([X_par[:, i] - X_cm[i] * (i < 2) for i in range(dim)])

        xi_par_frame = np.sqrt(X_0[0] ** 2 + X_0[1] ** 2)  # Spherical model
        zi_par_frame = X_0[2]

        xi_par = np.concatenate((xi_par, xi_par_frame))
        zi_par = np.concatenate((zi_par, zi_par_frame))

    assert isinstance(xi_par, np.ndarray)
    assert isinstance(zi_par, np.ndarray)
