from setuptools import setup, find_packages


setup(
    name="HydroAngleAnalyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add dependencies here, e.g., "numpy>=1.21.0"
    ],
    description="A simple Python library to parse MD trajectories from lammps and ASE and measure the contact through different methods",
    author="Gabriel",
    author_email="gabriel.taillandier@matgenix.com",
    url="https://github.com/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
