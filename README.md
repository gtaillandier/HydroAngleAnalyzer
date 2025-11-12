# HydroAngleAnalyzer

HydroAngleAnalyzer is a Python library designed to parse molecular dynamics (MD) trajectories from LAMMPS, ASE, or XYZ formats. Its objective is to provide a unified tool for referencing and implementing methods to measure contact angles with different approaches. Two methods implemented, for two models of droplet of liquids (spherical or cylindrical using PBC).

---

## Features

- **Flexible Trajectory Parsing:** Supports LAMMPS, ASE, and XYZ formats via a unified parser API.
- **Contact Angle Measurement:** Multiple methods (sliced, binned) for robust contact angle analysis.
- **Parallel & Batch Processing:** Efficient analysis of large trajectories.
- **Visualization:** Tools for plotting surfaces, angles, and analysis results.
- **Optional Dependencies:** `ovito` and `ase` are only required for specific workflows.

---

## Installation

### Prerequisites

Before installing HydroAngleAnalyzer, ensure you have the following prerequisites:

1. **Python 3.9 or higher**: Make sure you have Python 3.9 or higher installed on your system.
2. **Conda**: Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

### Optional install

If you need to analyze, study lammps trajectory the most convivinient is to install this option: 
```sh
pip install hydroangleanalyzer[ovito]
```
```sh
pip install hydroangleanalyzer[ase]
```
or to install all
```sh
pip install hydroangleanalyzer[all]
```

#### Install OVITO

OVITO must be installed using the following Conda command:

```sh
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
```
### Quick Start
Sliced Contact Angle (Parallel)

```python
from hydroangleanalyzer.contact_angle_method.sliced_method import ContactAngle_sliced_parallel
from hydroangleanalyzer.parser import DumpParser, Dump, Ase_Parser

# Prepare parser and water finder
parser = DumpParser('traj.lammpstrj', particle_type_wall={3})
wat_find = Dump_WaterMoleculeFinder('traj.lammpstrj', 
        particle_type_wall=['3'],
         oxygen_type=1,          
         hydrogen_type=2 )       
oxygen_indices = wat_find.get_water_oxygen_indices(num_frame=0)

# Initialize processor
processor = ContactAngle_sliced_parallel(
    filename='traj.lammpstrj',
    output_repo='results_postprocess/',
    delta_cylinder=5,
    type='cylinder_x',
    particle_type_wall={3},
    liquid_indices=oxygen_indices)

# Process frames in parallel
frames_to_process = list(range(1, 10))
contact_angles = processor.process_frames_parallel(frames_to_process, num_batches=2)
print("Mean contact angles:", contact_angles)
```
