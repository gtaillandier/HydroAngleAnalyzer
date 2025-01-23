# HydroAngleAnalyzer

HydroAngleAnalyzer is a simple Python library to parse MD trajectories from LAMMPS and ASE and measure the contact through different methods.

## Installation

### Prerequisites

Before installing HydroAngleAnalyzer, ensure you have the following prerequisites:

1. **Python 3.9 or higher**: Make sure you have Python 3.9 or higher installed on your system.
2. **Conda**: Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

### Install OVITO

OVITO must be installed using the following Conda command:

```sh
conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge ovito=3.11.3
