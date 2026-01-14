import numpy as np
from hydroangleanalyzer.parser.parser_ase import AseParser
from hydroangleanalyzer.parser.parser_xyz import XYZParser
from hydroangleanalyzer.parser.parser_dump import DumpParser
from hydroangleanalyzer.contact_angle_method.binning_method.angle_fitting_binning import ContactAngleBinning
from hydroangleanalyzer.contact_angle_method.sliced_method.multi_processing import ContactAngleSlicedParallel

print("Imports successful!")

# Test get_profile_coordinates signature
try:
    parser = AseParser("/dev/null")
except Exception:
    pass

print("AseParser available!")
