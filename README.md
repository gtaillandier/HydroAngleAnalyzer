# HydroAngleAnalyzer

HydroAngleAnalyzer provides modular tools to parse MD trajectories (LAMMPS dump, XYZ, ASE) and compute droplet contact angles using two complementary approaches:

1. Sliced Method (per-frame circle fit) – robust against transient shape changes.
2. Binning Density Method – averages frames into a density field for a single representative angle.


## Installation

### Prerequisites

Before installing HydroAngleAnalyzer, ensure you have the following prerequisites:

1. **Python 3.9 or higher**: Make sure you have Python 3.9 or higher installed on your system.
2. **Conda**: Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

Core (only to analyse simple xyz trajectories):

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

OVITO must be installed first in the conda environment and using the following Conda command:

```sh
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
```

## Quick Start


```python
from hydroangleanalyzer import (
    DumpParser, SlicedContactAngleAnalyzer, BinningContactAngleAnalyzer,
    detect_parser_type, contact_angle_analyzer
)

trajectory_file = "trajectory.lammpstrj"
parser = DumpParser(trajectory_file)

sliced = SlicedContactAngleAnalyzer(parser, output_repo="out_sliced", liquid_indices=oxygen_ids, droplet_geometry="spherical", delta_gamma=5)
results = sliced.analyze(frame_range=range(0, 50))
print(results["mean_angle"], results["std_angle"])

binning = BinningContactAngleAnalyzer(parser, output_dir="out_binned", liquid_indices=oxygen_ids, droplet_geometry="spherical")
results_binning = binning.analyze(frame_range=range(0, 200))
print(results_binning["mean_angle"], results_binning["std_angle"])
```
