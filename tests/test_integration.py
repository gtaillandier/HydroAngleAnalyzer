"""
Integration tests for hydro_angle_analyzer package.
These tests verify that the package works correctly with actual trajectory files.
"""

import os
import pytest
import numpy as np
from hydroangleanalyzer import FrameProcessor

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration

# Only run these tests if the trajectory file exists
needs_traj_file = pytest.mark.skipif(
    not os.path.exists('traj_10_3_330w_nve_4k_reajust.lammpstrj'),
    reason="Test trajectory file not found"
)

@needs_traj_file
def test_full_processing_flow(temp_output_dir):
    """
    Test the complete processing flow with a real trajectory file.
    This test processes multiple frames and checks the results.
    """
    frames_to_process = list(range(5))

    proc = FrameProcessor(
        filename='traj_10_3_330w_nve_4k_reajust.lammpstrj',
        output_repo=temp_output_dir,
        type='masspain',
        particle_type_wall={3},
        particule_liquid_type={1, 2}
    )

    results = proc.parallel_process_frames(frames_to_process)

    assert results is not None

    output_files = os.listdir(temp_output_dir)
    assert len(output_files) > 0

    txt_files = [f for f in output_files if f.endswith('.txt')]
    assert len(txt_files) > 0
    
    # If your processor creates specific output files, check for those
    # For example, if it creates CSV files:
    csv_files = [f for f in output_files if f.endswith('.csv')]
    assert len(csv_files) > 0

@needs_traj_file
def test_specific_frame_processing(temp_output_dir):
    """Test processing a specific frame and verify the results in detail."""
    # Process just frame 0
    frame_to_process = 0
    
    proc = FrameProcessor(
        filename='traj_10_3_330w_nve_4k_reajust.lammpstrj',
        output_repo=temp_output_dir,
        type='masspain'
    )
    
    # Process the single frame
    result = proc.parallel_process_frames([frame_to_process])
    
    # Detailed validation of the result
    # These assertions need to be adapted to your actual output format
    assert result is not None
    
    # Check if specific output files for frame 0 were created
    # For example:
    expected_files = [
        # List expected output files for frame 0
        # e.g., "frame_0_angles.csv", "frame_0_data.npy", etc.
    ]
    
    for expected_file in expected_files:
        if expected_file:  # Skip empty strings from the placeholder
            file_path = os.path.join(temp_output_dir, expected_file)
            assert os.path.exists(file_path), f"Expected output file {expected_file} not found"

@needs_traj_file
@pytest.mark.parametrize("processor_type", ["masspain", "mass", "pain"])
def test_different_processor_types(temp_output_dir, processor_type):
    """Test that different processor types work correctly."""
    proc = FrameProcessor(
        filename='traj_10_3_330w_nve_4k_reajust.lammpstrj',
        output_repo=temp_output_dir,
        type=processor_type
    )
    
    # Process one frame
    result = proc.parallel_process_frames([0])
    
    # Verify that processing completed successfully
    assert result is not None
    
    # Check that type-specific outputs were created
    # Add assertions specific to each processor type

@needs_traj_file
@pytest.mark.slow
def test_performance_with_multiple_frames(temp_output_dir):
    """Test performance with processing multiple frames."""
    # Process a larger number of frames to test performance
    frames_to_process = list(range(10))
    
    proc = FrameProcessor(
        filename='traj_10_3_330w_nve_4k_reajust.lammpstrj',
        output_repo=temp_output_dir,
        type='masspain'
    )
    
    # Measure execution time
    import time
    start_time = time.time()
    
    results = proc.parallel_process_frames(frames_to_process)
    
    execution_time = time.time() - start_time
    
    # Basic assertions
    assert results is not None
    
    # Performance threshold - adjust as needed
    assert execution_time < 60, f"Processing took too long: {execution_time:.2f} seconds"
    
    # Print performance info
    print(f"\nProcessed {len(frames_to_process)} frames in {execution_time:.2f} seconds")
    print(f"Average time per frame: {execution_time/len(frames_to_process):.2f} seconds")