import os
import pytest
import tempfile
import shutil

# Define fixture for trajectory file path
@pytest.fixture
def trajectory_file_path():
    """Return the path to the test trajectory file."""
    return 'traj_10_3_330w_nve_4k_reajust.lammpstrj'

# Define fixture for a temporary directory
@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

# If you need a small sample trajectory file for testing
@pytest.fixture
def sample_trajectory(trajectory_file_path, temp_output_dir):
    """
    Create a small sample trajectory file for testing.
    Only creates the sample if the source file exists.
    """
    if not os.path.exists(trajectory_file_path):
        pytest.skip(f"Source trajectory file '{trajectory_file_path}' not found")
    
    sample_path = os.path.join(temp_output_dir, "sample_trajectory.lammpstrj")
    
    # Read first few frames from the original file and write to sample
    try:
        with open(trajectory_file_path, 'r') as source, open(sample_path, 'w') as dest:
            # Copy the first 1000 lines or so (adjust based on your file format)
            for _ in range(1000):
                line = source.readline()
                if not line:
                    break
                dest.write(line)
    except Exception as e:
        pytest.skip(f"Failed to create sample trajectory: {e}")
    
    yield sample_path