import os
import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
from hydroangleanalyzer import DumpParser, ContactAngleAnalyzer

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def binning_params():
    """Default binning parameters for testing."""
    return {
        'xi_0': 0, 'xi_f': 100.0, 'nbins_xi': 50,
        'zi_0': 0.0, 'zi_f': 100.0, 'nbins_zi': 50
    }

@pytest.fixture
def analyzer(temp_dir, binning_params):
    """Fixture to create a ContactAngleAnalyzer instance with default parameters."""
    parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj', particle_type_wall={3})
    return ContactAngleAnalyzer(
        parser=parser,
        wall_height=4.89,
        type_model="masspain",
        width_masspain=21,
        binning_params=binning_params,
        output_dir=temp_dir
    )

needs_traj_file = pytest.mark.skipif(
    not os.path.exists('traj_10_3_330w_nve_4k_reajust.lammpstrj'),
    reason="Test trajectory file not found"
)

class TestContactAngleAnalyzer:
    @needs_traj_file
    def test_initialization(self, analyzer):
        """Test ContactAngleAnalyzer initialization."""
        assert analyzer is not None
        assert analyzer.wall_height == 4.89
        assert analyzer.type_model == "masspain"
        assert analyzer.width_masspain == 21
        assert analyzer.output_dir == temp_dir

    @needs_traj_file
    @pytest.mark.parametrize("batch_size", [3, 5, 10])
    def test_process_batch(self, analyzer, batch_size):
        """Test processing a single batch of frames."""
        angles = analyzer.process_batch(0, batch_size)
        assert angles is not None
        assert isinstance(angles, (np.ndarray, list))
        assert len(angles) > 0
        assert all(0 <= angle <= 180 for angle in angles)

    @needs_traj_file
    def test_process_all_batches(self, analyzer):
        """Test processing all batches."""
        angles = analyzer.process_all_batches(batch_size=5)
        assert angles is not None
        assert isinstance(angles, (np.ndarray, list))
        assert len(angles) > 0

        mean_angle = np.mean(angles)
        std_angle = np.std(angles)
        assert 0 <= mean_angle <= 180
        assert std_angle >= 0

    @needs_traj_file
    @pytest.mark.parametrize("model_type", ["masspain", "spherical"])
    def test_different_model_types(self, model_type, temp_dir, binning_params):
        """Test different model types."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        analyzer = ContactAngleAnalyzer(
            parser=parser,
            wall_height=4.89,
            type_model=model_type,
            width_masspain=21 if model_type == "masspain" else None,
            binning_params=binning_params,
            output_dir=temp_dir
        )

        angles = analyzer.process_batch(0, 3)
        assert angles is not None
        assert len(angles) > 0

    @needs_traj_file
    @pytest.mark.parametrize("binning", [
        {'xi_0': 0, 'xi_f': 100.0, 'nbins_xi': 25, 'zi_0': 0.0, 'zi_f': 100.0, 'nbins_zi': 25},
        {'xi_0': 10, 'xi_f': 90.0, 'nbins_xi': 50, 'zi_0': 10.0, 'zi_f': 90.0, 'nbins_zi': 50}
    ])
    def test_binning_parameter_variations(self, binning, temp_dir):
        """Test different binning parameters."""
        parser = DumpParser('traj_10_3_330w_nve_4k_reajust.lammpstrj')
        analyzer = ContactAngleAnalyzer(
            parser=parser,
            wall_height=4.89,
            type_model="masspain",
            width_masspain=21,
            binning_params=binning,
            output_dir=temp_dir
        )

        angles = analyzer.process_batch(0, 3)
        assert angles is not None
        assert len(angles) > 0

    def test_with_mock_parser(self, temp_dir, binning_params):
        """Test using a mocked parser to avoid file dependency."""
        mock_parser = MagicMock(spec=DumpParser)
        mock_frame = {
            'positions': np.random.rand(1000, 3) * 100,
            'types': np.random.randint(1, 4, 1000),
            'box': [0, 100, 0, 100, 0, 100]
        }

        mock_parser.read_frame.return_value = mock_frame
        mock_parser.read_frames.return_value = [mock_frame, mock_frame, mock_frame]
        mock_parser.count_frames.return_value = 100

        analyzer = ContactAngleAnalyzer(
            parser=mock_parser,
            wall_height=4.89,
            type_model="masspain",
            width_masspain=21,
            binning_params=binning_params,
            output_dir=temp_dir
        )

        angles = analyzer.process_batch(0, 3)
        mock_parser.read_frames.assert_called_once()
        assert angles is not None