import pytest
import numpy as np
import pathlib
from hydroangleanalyzer.parser import DumpParser, Dump_WaterMoleculeFinder
from hydroangleanalyzer.contact_angle_method import create_contact_angle_analyzer

# --- Fixtures ---
@pytest.fixture
def filename():
    # Use the correct path for your test file
    return pathlib.Path(__file__).parent.parent  / "trajectories" / "traj_10_3_330w_nve_4k_reajust.lammpstrj"

@pytest.fixture
def wat_find(filename):
    return Dump_WaterMoleculeFinder(
        filename,
        particle_type_wall={3},
        oxygen_type=1,
        hydrogen_type=2
    )

@pytest.fixture
def oxygen_indices(wat_find):
    return wat_find.get_water_oxygen_ids(num_frame=0)

@pytest.fixture
def parser(filename):
    return DumpParser(
        filename,
        particle_type_wall={3}
    )

@pytest.fixture
def binning_params():
    return {
        'xi_0': 0,
        'xi_f': 100.0,
        'nbins_xi': 50,
        'zi_0': 0.0,
        'zi_f': 100.0,
        'nbins_zi': 25
    }

# --- Unit Test for BinnedContactAngleAnalyzer ---
def test_binned_contact_angle_analyzer_with_real_data(filename, oxygen_indices, binning_params, tmp_path):
    # Use a temporary directory for output
    output_dir = tmp_path / "result_dump_masspain_noplot"

    # Create the analyzer
    analyzer = create_contact_angle_analyzer(
        method='binned',
        parser=DumpParser(filename, particle_type_wall={3}),
        output_dir=output_dir,
        liquid_indices=oxygen_indices,
        type_model='masspain_y',
        width_masspain=21,
        binning_params=binning_params,
        plot_graphs=False
    )

    # Run analysis for frame 1
    results = analyzer.analyze([1])

    # Assert results
    assert 'mean_angle' in results
    assert 'std_angle' in results
    assert 'angles' in results
    assert len(results['angles']) == 1
    assert 0 <= results['mean_angle'] <= 180
    assert np.isfinite(results['std_angle'])

# --- Optional: Test for multiple frames ---
def test_binned_contact_angle_analyzer_multiple_frames(filename, oxygen_indices, binning_params, tmp_path):
    output_dir = tmp_path / "result_dump_masspain_noplot_multiple"

    analyzer = create_contact_angle_analyzer(
        method='binned',
        parser=DumpParser(filename, particle_type_wall={3}),
        output_dir=output_dir,
        liquid_indices=oxygen_indices,
        type_model='masspain_y',
        width_masspain=21,
        binning_params=binning_params,
        plot_graphs=False
    )

    # Run analysis for frames 1, 2, and 3
    results = analyzer.analyze([1, 2, 3])

    # Assert results
    assert 'mean_angle' in results
    assert 'std_angle' in results
    assert 'angles' in results
    assert len(results['angles']) == 1
    assert 0 <= results['mean_angle'] <= 180    
    assert np.isfinite(results['std_angle'])
