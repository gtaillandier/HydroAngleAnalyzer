import os
import pytest
import numpy as np
from hydro_angle_analyzer import DumpParser

# Skip tests if the trajectory file doesn't exist
needs_traj_file = pytest.mark.skipif(
    not os.path.exists('traj_10_3_330w_nve_4k_reajust.lammpstrj'),
    reason="Test trajectory file not found"
)

class TestDumpParser:
    
    @needs_traj_file
    def test_initialization(self):
        """Test DumpParser initialization with existing file."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        assert parser is not None
        # Check that the parser has loaded the file
        assert hasattr(parser, 'file_path')
        assert parser.file_path == 'traj_10_3_330w_nve_4k_reajust.lammpstrj'
    
    def test_initialization_nonexistent_file(self):
        """Test DumpParser initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            DumpParser('nonexistent_file.lammpstrj')
    
    @needs_traj_file
    def test_read_frame(self):
        """Test reading a single frame from the trajectory."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        # Assuming the parser has a method to read a specific frame
        frame = parser.read_frame(0)  # Adjust method name as needed
        
        # Check that the frame data is valid
        assert frame is not None
        
        # Check frame structure (adjust based on actual return format)
        # For example, if it returns a dictionary with atom positions:
        if hasattr(frame, 'positions') or isinstance(frame, dict) and 'positions' in frame:
            positions = frame.positions if hasattr(frame, 'positions') else frame['positions']
            assert isinstance(positions, np.ndarray)
            assert positions.shape[1] == 3  # x, y, z coordinates
        
    @needs_traj_file
    def test_count_frames(self):
        """Test counting the total number of frames in the trajectory."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        # Assuming the parser has a method to count frames
        total_frames = parser.count_frames()  # Adjust method name as needed
        
        # Verify we have some frames
        assert total_frames > 0
        
        # If we know the expected number of frames, we can check that
        # assert total_frames == expected_value
    
    @needs_traj_file
    def test_read_multiple_frames(self):
        """Test reading multiple frames from the trajectory."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        # Assuming the parser has a method to read multiple frames
        frames = parser.read_frames([0, 1, 2])  # Adjust method name as needed
        
        # Check that we got the expected number of frames
        assert len(frames) == 3
        
        # Check that each frame has valid data
        for frame in frames:
            assert frame is not None
    
    @needs_traj_file
    def test_get_atom_types(self):
        """Test retrieving atom types from a frame."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        # Read a frame
        frame = parser.read_frame(0)
        
        # Assuming the parser has a method to get atom types from a frame
        atom_types = parser.get_atom_types(frame)  # Adjust method name as needed
        
        # Check that we got valid atom types
        assert atom_types is not None
        assert len(atom_types) > 0
        
        # If we know the expected atom types, we can check for them
        # assert set(atom_types) == {1, 2, 3}  # Example
    
    @needs_traj_file
    def test_get_box_dimensions(self):
        """Test retrieving box dimensions from a frame."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        # Read a frame
        frame = parser.read_frame(0)
        
        # Assuming the parser has a method to get box dimensions
        box_dims = parser.get_box_dimensions(frame)  # Adjust method name as needed
        
        # Check that we got valid box dimensions
        assert box_dims is not None
        
        # Check the structure of box dimensions (adjust based on actual format)
        # For example, if it returns [xlo, xhi, ylo, yhi, zlo, zhi]:
        assert len(box_dims) == 6
        assert box_dims[1] > box_dims[0]  # xhi > xlo
        assert box_dims[3] > box_dims[2]  # yhi > ylo
        assert box_dims[5] > box_dims[4]  # zhi > zlo