import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
from hydro_angle_analyzer import FrameProcessor, DumpParser, ContactAngleAnalyzer

# Path to test trajectory files - update these to point to your actual files
TEST_TRAJ_FILE = 'traj_10_3_330w_nve_4k_reajust.lammpstrj'

# Skip tests if the trajectory file doesn't exist
needs_traj_file = pytest.mark.skipif(
    not os.path.exists(TEST_TRAJ_FILE),
    reason=f"Test trajectory file '{TEST_TRAJ_FILE}' not found"
)

class TestFrameProcessor:
    
    def setup_method(self):
        """Setup for each test."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def teardown_method(self):
        """Teardown after each test."""
        self.temp_dir.cleanup()
    
    @needs_traj_file
    def test_init(self):
        """Test FrameProcessor initialization."""
        proc = FrameProcessor(
            filename=TEST_TRAJ_FILE,
            output_repo=self.temp_dir.name,
            type='masspain'
        )
        
        assert proc.filename == TEST_TRAJ_FILE
        assert proc.output_repo == self.temp_dir.name
        assert proc.type == 'masspain'
        
    @needs_traj_file
    def test_parallel_process_frames(self):
        """Test the parallel processing of frames."""
        frames_to_process = list(range(5))  # Process fewer frames for testing
        
        proc = FrameProcessor(
            filename=TEST_TRAJ_FILE,
            output_repo=self.temp_dir.name,
            type='masspain'
        )
        
        result = proc.parallel_process_frames(frames_to_process)
        
        # Validate the result (adjust based on your expected return values)
        assert result is not None
        
        # Check if any output files were created
        files = os.listdir(self.temp_dir.name)
        assert len(files) > 0
    
    @pytest.mark.parametrize("frame_type", ["masspain", "mass", "pain"])
    @needs_traj_file
    def test_different_types(self, frame_type):
        """Test using different types of frame processors."""
        frames_to_process = [0]  # Just process one frame for testing
        
        proc = FrameProcessor(
            filename=TEST_TRAJ_FILE,
            output_repo=self.temp_dir.name,
            type=frame_type
        )
        
        result = proc.parallel_process_frames(frames_to_process)
        assert result is not None
    
    @patch('hydro_angle_analyzer.FrameProcessor._process_single_frame')
    def test_mock_processing(self, mock_process):
        """Test with mocked frame processing to avoid file requirements."""
        # Set up mock return value
        mock_process.return_value = {"mock": "data"}
        
        proc = FrameProcessor(
            filename="mock_file.lammpstrj",
            output_repo=self.temp_dir.name,
            type='masspain'
        )
        
        frames = [0, 1, 2]
        result = proc.parallel_process_frames(frames)
        
        # Check that _process_single_frame was called for each frame
        assert mock_process.call_count == len(frames)
        
    def test_invalid_file(self):
        """Test behavior with non-existent file."""
        with pytest.raises(FileNotFoundError):
            proc = FrameProcessor(
                filename="non_existent_file.lammpstrj",
                output_repo=self.temp_dir.name,
                type='masspain'
            )
            proc.parallel_process_frames([0])


class TestIntegration:
    """Integration tests for all components together."""
    
    def setup_method(self):
        """Setup for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Default binning parameters
        self.binning_params = {
            'xi_0': 0, 'xi_f': 100.0, 'nbins_xi': 50,
            'zi_0': 0.0, 'zi_f': 100.0, 'nbins_zi': 50
        }
        
        self.wall_height = 4.89
        self.type_model = "masspain"
        self.width_masspain = 21
    
    def teardown_method(self):
        """Teardown after each test."""
        self.temp_dir.cleanup()
    
    @needs_traj_file
    def test_full_workflow(self):
        """Test the full workflow from parsing to angle analysis."""
        # Initialize parser
        parser = DumpParser(TEST_TRAJ_FILE)
        
        # Initialize analyzer
        analyzer = ContactAngleAnalyzer(
            parser=parser,
            wall_height=self.wall_height,
            type_model=self.type_model,
            width_masspain=self.width_masspain,
            binning_params=self.binning_params,
            output_dir=self.temp_dir.name
        )
        
        # Process with a small batch size
        batch_size = 5
        angles = analyzer.process_all_batches(batch_size=batch_size)
        
        # Check results
        assert angles is not None
        assert len(angles) > 0
        
        # Calculate statistics (as in the example script)
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        print(f"Average contact angle: {mean_angle:.2f} ± {std_angle:.2f}")
        
        # Check that output files were created
        files = os.listdir(self.temp_dir.name)
        assert len(files) > 0
    
    @needs_traj_file
    def test_frameprocessor_with_dumpparser(self):
        """Test using FrameProcessor with DumpParser."""
        # First parse the trajectory
        parser = DumpParser(TEST_TRAJ_FILE)
        
        # Then use FrameProcessor to process frames
        proc = FrameProcessor(
            filename=TEST_TRAJ_FILE,
            output_repo=self.temp_dir.name,
            type='masspain'
        )
        
        # Process a few frames
        frames_to_process = list(range(3))
        result = proc.parallel_process_frames(frames_to_process)
        
        assert result is not None
        
        # Check that output files were created
        files = os.listdir(self.temp_dir.name)
        assert len(files) > 0


if __name__ == "__main__":
    pytest.main(["-v"])