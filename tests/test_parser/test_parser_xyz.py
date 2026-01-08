import os

import numpy as np
import pytest

from hydroangleanalyzer.parser.parser_xyz import XYZParser

# Path to the test trajectory file
TRAJECTORY_PATH = os.path.join(
    os.path.dirname(__file__), "../trajectories/slice_10_mace_mlips_cylindrical_2_5.xyz"
)


# --- Fixture for XYZ_Parser ---
@pytest.fixture
def xyz_parser():
    return XYZParser(TRAJECTORY_PATH)


# --- Test load_xyz_file ---
def test_load_xyz_file(xyz_parser):
    frames = xyz_parser.frames
    assert len(frames) > 0  # At least one frame should be loaded
    for frame in frames:
        assert "symbols" in frame
        assert "positions" in frame
        assert "lattice_matrix" in frame
        assert isinstance(frame["symbols"], np.ndarray)
        assert isinstance(frame["positions"], np.ndarray)
        assert frame["lattice_matrix"].shape == (3, 3)


# --- Test parse ---
def test_parse(xyz_parser):
    frame_indexs = 0
    positions = xyz_parser.parse(frame_indexs)
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 3  # x, y, z coordinates

    # Test with specific indices
    indices = [0, 1, 2]
    positions_subset = xyz_parser.parse(frame_indexs, indices)
    assert positions_subset.shape[0] == len(indices)


# --- Test parse_liquid_particles ---
def test_parse_liquid_particles(xyz_parser):
    frame_indexs = 0
    particle_type_liquid = ["O", "H"]
    liquid_positions = xyz_parser.parse_liquid_particles(
        particle_type_liquid, frame_indexs
    )
    assert isinstance(liquid_positions, np.ndarray)
    assert liquid_positions.shape[1] == 3  # x, y, z coordinates


# --- Test get_cylindrical_coordinates ---
def test_get_cylindrical_coordinates(xyz_parser):
    frame_list = [0, 1]
    xi_par, zi_par, frame_indexss = xyz_parser.get_cylindrical_coordinates(frame_list)
    assert isinstance(xi_par, np.ndarray)
    assert isinstance(zi_par, np.ndarray)
    assert frame_indexss == len(frame_list)
    assert xi_par.shape == zi_par.shape

    # Test with liquid_indices
    liquid_indices = [0, 1, 2]
    xi_par, zi_par, _ = xyz_parser.get_cylindrical_coordinates(
        frame_list, liquid_indices=liquid_indices
    )
    assert xi_par.size > 0
    assert zi_par.size > 0


# --- Test box_length_max ---
def test_box_length_max(xyz_parser):
    frame_indexs = 0
    max_length = xyz_parser.box_length_max(frame_indexs)
    assert isinstance(max_length, float)
    assert max_length > 0


# --- Test box_size_x and box_size_y ---
def test_box_size_x(xyz_parser):
    frame_indexs = 0
    box_size_x = xyz_parser.box_size_x(frame_indexs)
    assert isinstance(box_size_x, float)
    assert box_size_x > 0


def test_box_size_y(xyz_parser):
    frame_indexs = 0
    box_size_y = xyz_parser.box_size_y(frame_indexs)
    assert isinstance(box_size_y, float)
    assert box_size_y > 0


# --- Test frame_count ---
def test_frame_count(xyz_parser):
    total_frames = xyz_parser.frame_count()
    assert isinstance(total_frames, int)
    assert total_frames > 0
