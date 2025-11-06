# Tutorial: Contact Angle Analysis (Binned Method)

This tutorial demonstrates how to compute the contact angle using the **binned method** in `hydroangleanalyzer`.  
The method divides the simulation box into spatial bins to calculate the liquid–solid interface and the corresponding contact angle.

---

## 1. Overview

The **binned method** works by:
1. Extracting the positions of water molecules (typically oxygen atoms).
2. Dividing the region of interest into bins in the **x–z** plane.
3. Computing density profiles and fitting the interface shape.
4. Deriving the contact angle from the interface curvature.

---

## 2. Prerequisites

Make sure your Python environment has:
```bash
pip install hydroangleanalyzer ase numpy
```

Your trajectory file (e.g., a LAMMPS dump file) should contain:
- Atom IDs, types, and positions
- Wall atoms (solid surface)
- Water molecules (O and H atoms)

Example trajectory:
```
tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj
```

---

## 3. Example Script
```python
# Import necessary modules
from hydroangleanalyzer.parser import DumpParser, Dump_WaterMoleculeFinder
from hydroangleanalyzer.contact_angle_method import create_contact_angle_analyzer

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/traj_10_3_330w_nve_4k_reajust.lammpstrj"

# --- Step 2: Initialize the water molecule finder ---
# This identifies O and H atoms in water molecules
wat_find = Dump_WaterMoleculeFinder(
    filename,
    particle_type_wall={3},  # Wall atom types
    oxygen_type=1,           # Oxygen atom type
    hydrogen_type=2          # Hydrogen atom type
)

# --- Step 3: Get oxygen atom indices for the first frame ---
oxygen_indices = wat_find.get_water_oxygen_ids(num_frame=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 4: Define binning parameters ---
binning_params = {
    'xi_0': 0.0,       # Minimum x-coordinate
    'xi_f': 100.0,     # Maximum x-coordinate
    'nbins_xi': 50,    # Number of bins along x
    'zi_0': 0.0,       # Minimum z-coordinate
    'zi_f': 100.0,     # Maximum z-coordinate
    'nbins_zi': 25     # Number of bins along z
}

# --- Step 5: Initialize the parser ---
parser = DumpParser(filename)

# --- Step 6: Create the contact angle analyzer ---
analyzer = create_contact_angle_analyzer(
    method='binned',
    parser=parser,
    output_dir='results_binned_example',
    liquid_indices=oxygen_indices,
    type_model='cylinder_y',     # Interface fitting model
    width_cylinder=21,           # Width parameter for interface fit
    binning_params=binning_params,
    plot_graphs=False            # Disable plotting for automated runs
)

# --- Step 7: Run analysis for a frame range ---
results = analyzer.analyze([1])  # Analyze frame 1
print("Analysis results:", results)
```

---

## 4. Output

Running this example will:
- Parse the trajectory
- Compute the interface shape and local contact angle
- Save results (if enabled) under `results_binned_example/`

Example printed output:
```
Number of water molecules: 329
Analysis results: {'frame': 1, 'contact_angle': 101.3, 'fit_quality': 0.97}
```

---

## 5. Tips

- Adjust `xi_f`, `zi_f`, and the bin counts (`nbins_xi`, `nbins_zi`) according to your simulation box dimensions.
- If the wall surface is not flat or the system is tilted, pre-align it before analysis.
- Use `plot_graphs=True` to visualize the binned density and interface fitting.
- For multiple frames: `analyzer.analyze(range(0, 100, 10))`.

---
