import os
import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
from hydro_angle_analyzer import DumpParser, ContactAngleAnalyzer

# Skip tests if the trajectory file doesn't exist
needs_traj_file = pytest.mark.skipif(
    not os.path.exists('traj_10_3_330w_nve_4k_reajust.lammpstrj'),
    reason="Test trajectory file not found"
)

class TestContactAngleAnalyzer:
    
    def setup_method(self):
        """Setup for each test."""
        # Create a temporary directory for test outputs
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
    def test_initialization(self):
        """Test ContactAngleAnalyzer initialization."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        analyzer = ContactAngleAnalyzer(
            parser=parser,
            wall_height=self.wall_height,
            type_model=self.type_model,
            width_masspain=self.width_masspain,
            binning_params=self.binning_params,
            output_dir=self.temp_dir.name
        )
        
        # Check that the analyzer was initialized correctly
        assert analyzer is not None
        assert analyzer.parser == parser
        assert analyzer.wall_height == self.wall_height
        assert analyzer.type_model == self.type_model
        assert analyzer.width_masspain == self.width_masspain
        assert analyzer.binning_params == self.binning_params
        assert analyzer.output_dir == self.temp_dir.name
    
    @needs_traj_file
    def test_process_batch(self):
        """Test processing a single batch of frames."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        analyzer = ContactAngleAnalyzer(
            parser=parser,
            wall_height=self.wall_height,
            type_model=self.type_model,
            width_masspain=self.width_masspain,
            binning_params=self.binning_params,
            output_dir=self.temp_dir.name
        )
        
        # Process a small batch (adapt method name if needed)
        batch_size = 5
        batch_index = 0
        angles = analyzer.process_batch(batch_index, batch_size)
        
        # Check that we got valid angles
        assert angles is not None
        assert isinstance(angles, np.ndarray) or isinstance(angles, list)
        assert len(angles) > 0
        
        # Check that all angles are within a reasonable range (0-180 degrees)
        for angle in angles:
            assert 0 <= angle <= 180
        
        # Check that output files were created
        files = os.listdir(self.temp_dir.name)
        assert len(files) > 0
    
    @needs_traj_file
    def test_process_all_batches(self):
        """Test processing all batches."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        analyzer = ContactAngleAnalyzer(
            parser=parser,
            wall_height=self.wall_height,
            type_model=self.type_model,
            width_masspain=self.width_masspain,
            binning_params=self.binning_params,
            output_dir=self.temp_dir.name
        )
        
        # Process all batches with a small batch size
        batch_size = 5
        angles = analyzer.process_all_batches(batch_size=batch_size)
        
        # Check that we got valid angles
        assert angles is not None
        assert isinstance(angles, np.ndarray) or isinstance(angles, list)
        assert len(angles) > 0
        
        # Check statistics
        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        
        assert 0 <= mean_angle <= 180
        assert std_angle >= 0
        
        # Check output files
        files = os.listdir(self.temp_dir.name)
        assert len(files) > 0
    
    @needs_traj_file
    @pytest.mark.parametrize("model_type", ["masspain", "spherical"])
    def test_different_model_types(self, model_type):
        """Test different model types."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        analyzer = ContactAngleAnalyzer(
            parser=parser,
            wall_height=self.wall_height,
            type_model=model_type,
            width_masspain=self.width_masspain if model_type == "masspain" else None,
            binning_params=self.binning_params,
            output_dir=self.temp_dir.name
        )
        
        # Process a small batch
        batch_size = 3
        angles = analyzer.process_batch(0, batch_size)
        
        # Check that we got valid angles
        assert angles is not None
        assert len(angles) > 0
    
    @needs_traj_file
    def test_binning_parameter_variations(self):
        """Test different binning parameters."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        
        # Test with different binning parameters
        binning_variations = [
            {
                'xi_0': 0, 'xi_f': 100.0, 'nbins_xi': 25,  # Fewer bins
                'zi_0': 0.0, 'zi_f': 100.0, 'nbins_zi': 25
            },
            {
                'xi_0': 10, 'xi_f': 90.0, 'nbins_xi': 50,  # Different range
                'zi_0': 10.0, 'zi_f': 90.0, 'nbins_zi': 50
            }
        ]
        
        for binning in binning_variations:
            analyzer = ContactAngleAnalyzer(
                parser=parser,
                wall_height=self.wall_height,
                type_model=self.type_model,
                width_masspain=self.width_masspain,
                binning_params=binning,
                output_dir=self.temp_dir.name
            )
            
            # Process a small batch
            angles = analyzer.process_batch(0, 3)
            
            # Check that we got valid angles
            assert angles is not None
            assert len(angles) > 0
    
    def test_with_mock_parser(self):
        """Test using a mocked parser to avoid file dependency."""
        # Create a mock parser
        mock_parser = MagicMock(spec=DumpParser)
        
        # Set up mock behavior for frame reading
        mock_frame = {
            'positions': np.random.rand(1000, 3) * 100,  # Random positions
            'types': np.random.randint(1, 4, 1000),      # Random atom types
            'box': [0, 100, 0, 100, 0, 100]              # Box dimensions
        }
        
        # Configure the mock to return our fake frame
        mock_parser.read_frame.return_value = mock_frame
        mock_parser.read_frames.return_value = [mock_frame, mock_frame, mock_frame]
        mock_parser.count_frames.return_value = 100
        
        # Create analyzer with mock parser
        analyzer = ContactAngleAnalyzer(
            parser=mock_parser,
            wall_height=self.wall_height,
            type_model=self.type_model,
            width_masspain=self.width_masspain,
            binning_params=self.binning_params,
            output_dir=self.temp_dir.name
        )
        
        # Process a small batch
        angles = analyzer.process_batch(0, 3)
        
        # Verify the mock was called correctly
        mock_parser.read_frames.assert_called_once()
        
        # Check results
        assert angles is not None