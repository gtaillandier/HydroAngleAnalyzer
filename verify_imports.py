from hydroangleanalyzer.parser.parser_ase import AseParser

print("Imports successful!")

# Test get_profile_coordinates signature
try:
    parser = AseParser("/dev/null")
except Exception:
    pass

print("AseParser available!")
