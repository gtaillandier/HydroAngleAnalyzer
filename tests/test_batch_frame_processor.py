import os
import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock
from hydroangleanalyzer import BatchFrameProcessor, WaterOxygenDumpParser, DumpParse_wall

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def batch_frame_processor(temp_dir):
    """Fixture to create a BatchFrameProcessor instance with default parameters."""
    return BatchFrameProcessor(
        filename='traj_10_3_330w_nve_4k_reajust.lammpstrj',
        output_repo=temp_dir,
        type='masspain',
        particle_type_wall={3},
        particule_liquid_type={1, 2},
        oxygen_type=1,
        hydrogen_type=2
    )

needs_traj_file = pytest.mark.skipif(
    not os.path.exists('traj_10_3_330w_nve_4k_reajust.lammpstrj'),
    reason="Test trajectory file not found"
)

class TestBatchFrameProcessor:
    @needs_traj_file
    def test_initialization(self, batch_frame_processor, temp_dir):
        """Test BatchFrameProcessor initialization."""
        assert batch_frame_processor is not None
        assert batch_frame_processor.filename == 'traj_10_3_330w_nve_4k_reajust.lammpstrj'
        assert batch_frame_processor.output_repo == temp_dir
        assert batch_frame_processor.particle_type_wall == {1}
        assert batch_frame_processor.particule_liquid_type == {2, 3}

    @pytest.mark.parametrize("frames, num_batches, expected_batch_count", [
        ([1, 2, 3, 4, 5], 2, 2),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, 3),
        ([1, 2, 3], 5, 3)  # More batches than frames
    ])
    def test_create_batches(self, batch_frame_processor, frames, num_batches, expected_batch_count):
        """Test the _create_batches method."""
        batches = batch_frame_processor._create_batches(frames, num_batches)
        assert len(batches) == expected_batch_count
        assert sum(len(batch) for batch in batches) == len(frames)

    @needs_traj_file
    @patch("hydro_angle_analyzer.WaterOxygenDumpParser")
    @patch("hydro_angle_analyzer.DumpParse_wall")
    def test_process_batch_worker(self, mock_wall_parser, mock_water_parser, batch_frame_processor):
        """Test the _process_batch_worker method."""
        mock_water_parser_instance = MagicMock()
        mock_water_parser.return_value = mock_water_parser_instance
        mock_wall_parser_instance = MagicMock()
        mock_wall_parser.return_value = mock_wall_parser_instance

        mock_water_parser_instance.get_water_oxygen_positions.return_value = np.random.rand(100, 3)
        mock_wall_parser_instance.find_highest_wall_part.return_value = 5.0

        batch_frames = [1, 2, 3]
        results = batch_frame_processor._process_batch_worker(batch_frames)
        assert len(results) == len(batch_frames)
        assert all(isinstance(result, tuple) for result in results)

    @needs_traj_file
    def test_process_frames_parallel(self, batch_frame_processor):
        """Test the process_frames_parallel method."""
        frames = list(range(1, 11))
        results = batch_frame_processor.process_frames_parallel(frames, num_batches=2, max_workers=2)
        assert len(results) == len(frames)
        assert all(isinstance(frame, int) and isinstance(angle, float) for frame, angle in results.items())

    @needs_traj_file
    def test_get_batch_info(self, batch_frame_processor):
        """Test the get_batch_info method."""
        frames = list(range(1, 101))
        num_batches = 5
        batch_info = batch_frame_processor.get_batch_info(frames, num_batches)
        assert batch_info["total_frames"] == len(frames)
        assert batch_info["num_batches"] == num_batches
        assert len(batch_info["batch_sizes"]) == num_batches
