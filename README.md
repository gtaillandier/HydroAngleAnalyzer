# HydroAngleAnalyzer

HydroAngleAnalyzer provides modular tools to parse MD trajectories (LAMMPS dump, XYZ, ASE) and compute droplet contact angles using two complementary approaches:

1. Sliced Method (per-frame circle fit) – robust against transient shape changes.
2. Binning Density Method – averages frames into a density field for a single representative angle.


## Installation

### Prerequisites

Before installing HydroAngleAnalyzer, ensure you have the following prerequisites:

1. **Python 3.9 or higher**: Make sure you have Python 3.9 or higher installed on your system.
2. **Conda**: Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

Core (no optional heavy deps):

```bash
pip install hydroangleanalyzer
```

With OVITO:
```bash
pip install hydroangleanalyzer[ovito]
```
With ASE:
```bash
pip install hydroangleanalyzer[ase]
```
All optional:
```bash
pip install hydroangleanalyzer[all]
```

#### Install OVITO

OVITO must be installed using the following Conda command:

```sh
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
```

## Quick Start


```python
from hydroangleanalyzer import (
    DumpParser, SlicedContactAngleAnalyzer, BinnedContactAngleAnalyzer,
    detect_parser_type, contact_angle_analyzer
)

traj = "traj.lammpstrj"
parser = DumpParser(traj)  # requires ovito extra
oxygen_ids = [/* obtain water oxygen IDs via Dump_WaterMoleculeFinder */]

sliced = SlicedContactAngleAnalyzer(parser, output_repo="out_sliced", liquid_indices=oxygen_ids, droplet_geometry="spherical", delta_gamma=5)
res = sliced.analyze(frame_range=range(0, 50))
print(res["mean_angle"], res["std_angle"])  # per-frame distribution

binning = BinnedContactAngleAnalyzer(parser, output_dir="out_binned", liquid_indices=oxygen_ids, droplet_geometry="spherical")
res_b = binning.analyze(frame_range=range(0, 200))
print(res_b["mean_angle"], res_b["std_angle"])  # single or batched average
```

## Visualization

```python
from hydroangleanalyzer import SlicedTrajectoryAnalyzer, Droplet_sliced_Plotter, plot_liquid_particles

analyzer = SlicedTrajectoryAnalyzer(["out_sliced"], time_steps={"out_sliced":0.5})
analyzer.read_data()
analyzer.plot_median_alfas_evolution("median_angles.png")

# Per frame droplet view
import numpy as np
oxygen_positions = np.loadtxt("oxygen_frame0.txt")  # example file you generate
surfaces = np.load("out_sliced/surfacesframe0.npy", allow_pickle=True)
popts = np.load("out_sliced/poptsframe0.npy")  # [Xc, Zc, R, z_cut]
wall_positions = np.loadtxt("wall_atoms.txt")
plotter = Droplet_sliced_Plotter()
plotter.plot_surface_points(oxygen_positions, surfaces, popts[0], wall_positions, "droplet_frame0.png", alpha=75)
```

## Troubleshooting

NaN angles: Usually occur when the surface filter removes too many points (empty slice). Adjust `surface_filter_offset` (default 2.0) in `ContactAngle_sliced` or relax slice width. Ensure enough atoms remain after filtering (>=3) for circle fitting.

Empty outputs / NoneType failures: Confirm `width_cylinder` and `delta_cylinder` are passed for cylindrical models and `delta_gamma` for spherical model. Parser must supply box dimensions for automatic max distance estimation.

Multiprocessing hangs: Use the batch-parallel wrapper (`ContactAngle_sliced_parallel.process_frames_parallel`) which employs spawn start method; avoid invoking OVITO parsers inside global contexts before multiprocessing starts.

OVITO ImportError: Install with the ovito extra or via the Conda command listed above. Verify channel priority and version pin if dependency resolution fails.

## Optional Dependencies Strategy
OVITO and ASE are only imported inside the respective parser classes. Installing the package without extras keeps dependencies minimal. Calling an OVITO/ASE parser without installing raises a clear ImportError with installation instructions.

## Docstring Style

NumPy-style docstrings adopted across modules. Pre-commit enforces pydocstyle with selected rules; legacy underscores (e.g. `box_length_max`) retained with modern alias for backward compatibility.

## Oxygen ID Retrieval Example

```python
from hydroangleanalyzer import Dump_WaterMoleculeFinder

finder = Dump_WaterMoleculeFinder("traj.lammpstrj", particle_type_wall={3}, oxygen_type=1, hydrogen_type=2)
oxygen_ids = finder.get_water_oxygen_ids(frame_indexs=0)
```

Use these indices for sliced or binning analyzers via `liquid_indices=oxygen_ids`.

## Public API
Explicit exports avoid wildcard imports and improve static analysis. See `hydroangleanalyzer/__init__.py` for the authoritative list.

## Upcoming BaseAngleMethod

A common abstract base will standardize interface (`analyze`, `predict_contact_angle`, metadata access) between sliced and binning strategies enabling factory dispatch and future auto-discovery of new angle methods.

## Contributing
Run pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

Style: black (line length 88 initially), isort (black profile), flake8, pydocstyle.

## Roadmap
- Integrate BaseAngleMethod for unified interface.
- Extend visualization for interactive time sliders (Plotly).
- Add CLI entry points for batch processing.

## License
MIT
