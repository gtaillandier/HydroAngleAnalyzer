import os

import numpy as np
import pytest

from hydroangleanalyzer.parser.parser_ase import AseParser

# Path to the test trajectory file (ASE format)
TRAJECTORY_PATH = os.path.join(
    os.path.dirname(__file__),
    "../trajectories/slice_10_mace_mlips_cylindrical_2_5.traj",
)


# --- Fixture for Ase_Parser ---
@pytest.fixture
def ase_parser():
    return AseParser(TRAJECTORY_PATH)


# --- Test parse ---
def test_parse(ase_parser):
    frame_indexs = 0
    positions = ase_parser.parse(frame_indexs)
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 3  # x, y, z coordinates

    # Test with specific indices
    indices = [0, 1, 2]
    positions_subset = ase_parser.parse(frame_indexs, indices)
    assert positions_subset.shape[0] == len(indices)


# --- Test parse_liquid_particles ---
def test_parse_liquid_particles(ase_parser):
    frame_indexs = 0
    particle_type_liquid = ["O", "H"]
    liquid_positions = ase_parser.parse_liquid_particles(
        particle_type_liquid, frame_indexs
    )
    assert isinstance(liquid_positions, np.ndarray)
    assert liquid_positions.shape[1] == 3  # x, y, z coordinates


# --- Test get_cylindrical_coordinates ---
def test_get_cylindrical_coordinates(ase_parser, capsys):
    frame_list = [0, 1]
    xi_par, zi_par, frame_indexss = ase_parser.get_cylindrical_coordinates(frame_list)
    assert isinstance(xi_par, np.ndarray)
    assert isinstance(zi_par, np.ndarray)
    assert frame_indexss == len(frame_list)
    assert xi_par.shape == zi_par.shape

    # Test with liquid_indices
    liquid_indices = [0, 1, 2]
    xi_par, zi_par, _ = ase_parser.get_cylindrical_coordinates(
        frame_list, liquid_indices=liquid_indices
    )
    assert xi_par.size > 0
    assert zi_par.size > 0

    # Check print output
    captured = capsys.readouterr()
    assert "xi range:" in captured.out
    assert "zi range:" in captured.out


# --- Test box_size_x and box_size_y ---
def test_box_size_x(ase_parser, capsys):
    frame_indexs = 0
    box_size_x = ase_parser.box_size_x(frame_indexs)
    assert isinstance(box_size_x, float)
    assert box_size_x > 0


def test_box_size_y(ase_parser):
    frame_indexs = 0
    box_size_y = ase_parser.box_size_y(frame_indexs)
    assert isinstance(box_size_y, float)
    assert box_size_y > 0


# --- Test box_length_max ---
def test_box_length_max(ase_parser):
    frame_indexs = 0
    max_length = ase_parser.box_length_max(frame_indexs)
    assert isinstance(max_length, float)
    assert max_length > 0


# --- Test frame_count ---
def test_frame_count(ase_parser):
    total_frames = ase_parser.frame_count()
    assert isinstance(total_frames, int)
    assert total_frames > 0


# --- Test type_model in get_cylindrical_coordinates ---
def test_get_cylindrical_coordinates_type_model(ase_parser):
    frame_list = [0]
    xi_par, zi_par, _ = ase_parser.get_cylindrical_coordinates(
        frame_list, type_model="spherical"
    )
    assert isinstance(xi_par, np.ndarray)
    assert isinstance(zi_par, np.ndarray)
