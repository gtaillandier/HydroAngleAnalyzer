import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from hydroangleanalyzer import DumpParser

@pytest.fixture
def dump_parser():
    """Fixture to create a DumpParser instance with the test trajectory file."""
    return DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj', particle_type_wall={3})

needs_traj_file = pytest.mark.skipif(
    not os.path.exists('traj_10_3_330w_nve_4k_reajust.lammpstrj'),
    reason="Test trajectory file not found"
)

class TestDumpParser:
    @needs_traj_file
    def test_initialization(self, dump_parser):
        """Test DumpParser initialization with an existing file."""
        assert dump_parser is not None
        assert hasattr(dump_parser, 'file_path')
        assert dump_parser.file_path == 'traj_10_3_330w_nve_4k_reajust.lammpstrj'

    def test_initialization_nonexistent_file(self):
        """Test DumpParser initialization with a non-existent file."""
        with pytest.raises(FileNotFoundError):
            DumpParser('nonexistent_file.lammpstrj')

    @needs_traj_file
    @pytest.mark.parametrize("frame_index", [0, 1, 2])
    def test_read_frame(self, dump_parser, frame_index):
        """Test reading a single frame from the trajectory."""
        frame = dump_parser.read_frame(frame_index)
        assert frame is not None

        if hasattr(frame, 'positions') or (isinstance(frame, dict) and 'positions' in frame):
            positions = frame.positions if hasattr(frame, 'positions') else frame['positions']
            assert isinstance(positions, np.ndarray)
            assert positions.shape[1] == 3  # x, y, z coordinates

    @needs_traj_file
    def test_count_frames(self, dump_parser):
        """Test counting the total number of frames in the trajectory."""
        total_frames = dump_parser.count_frames()
        assert total_frames > 0

    @needs_traj_file
    @pytest.mark.parametrize("frame_indices", [[0, 1, 2], [5, 10, 15]])
    def test_read_multiple_frames(self, dump_parser, frame_indices):
        """Test reading multiple frames from the trajectory."""
        frames = dump_parser.read_frames(frame_indices)
        assert len(frames) == len(frame_indices)

        for frame in frames:
            assert frame is not None

    @needs_traj_file
    def test_get_atom_types(self, dump_parser):
        """Test retrieving atom types from a frame."""
        frame = dump_parser.read_frame(0)
        atom_types = dump_parser.get_atom_types(frame)
        assert atom_types is not None
        assert len(atom_types) > 0

    @needs_traj_file
    def test_get_box_dimensions(self, dump_parser):
        """Test retrieving box dimensions from a frame."""
        frame = dump_parser.read_frame(0)
        box_dims = dump_parser.get_box_dimensions(frame)
        assert box_dims is not None
        assert len(box_dims) == 6
        assert box_dims[1] > box_dims[0]  # xhi > xlo
        assert box_dims[3] > box_dims[2]  # yhi > ylo
        assert box_dims[5] > box_dims[4]  # zhi > zlo

    def test_read_frame_with_mock(self):
        """Test reading a frame using a mocked file."""
        mock_parser = MagicMock(spec=DumpParser)
        mock_frame = {
            'positions': np.random.rand(100, 3) * 100,
            'types': np.random.randint(1, 4, 100),
            'box': [0, 100, 0, 100, 0, 100]
        }
        mock_parser.read_frame.return_value = mock_frame

        frame = mock_parser.read_frame(0)
        assert frame is not None
        assert isinstance(frame['positions'], np.ndarray)
        assert frame['positions'].shape[1] == 3