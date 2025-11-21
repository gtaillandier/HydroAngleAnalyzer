# Import necessary modules
from hydroangleanalyzer.contact_angle_method import contact_angle_analyzer
from hydroangleanalyzer.parser import Dump_WaterMoleculeFinder, DumpParser

# --- Step 1: Define the trajectory file ---
filename = "../../tests/trajectories/traj_spherical_drop_4k.lammpstrj"

# --- Step 2: Initialize the water molecule finder ---
# This identifies O and H atoms in water molecules
wat_find = Dump_WaterMoleculeFinder(
    filename,
    particle_type_wall={3},  # Wall atom types
    oxygen_type=1,  # Oxygen atom type
    hydrogen_type=2,  # Hydrogen atom type
)

# --- Step 3: Get oxygen atom indices for the first frame ---
oxygen_indices = wat_find.get_water_oxygen_ids(num_frame=0)
print("Number of water molecules:", len(oxygen_indices))

# --- Step 4: Define binning parameters ---
binning_params = {
    "xi_0": 0.0,  # Minimum x-coordinate
    "xi_f": 70.0,  # Maximum x-coordinate
    "nbins_xi": 30,  # Number of bins along x
    "zi_0": 0.0,  # Minimum z-coordinate
    "zi_f": 70.0,  # Maximum z-coordinate
    "nbins_zi": 30,  # Number of bins along z
}

# --- Step 5: Initialize the parser ---
parser = DumpParser(filename)

# --- Step 6: Create the contact angle analyzer ---
analyzer = contact_angle_analyzer(
    method="binned",
    parser=parser,
    output_dir="results_binned_example",
    liquid_indices=oxygen_indices,
    type_model="spherical",  # Interface fitting model
    binning_params=binning_params,
    plot_graphs=True,  # Enable plotting for automated runs
)

# --- Step 7: Run analysis for a frame range ---
results = analyzer.analyze([1])  # Analyze frame 1
print("Analysis results:", results)
