---
title: 'Wetting-angle-kit: a Python package to streamline the computation of wetting angles of nanoparticles in liquids'
tags:
  - Python
  - Nanodroplets
  - Molecular dynamics
  - Wetting properties
authors:
  - name: Gabriel Taillandier
    orcid: 0009-0006-9544-0982
    affiliation: "1, 2"
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:

affiliations:
 - name: Matgenix, A6K Advanced Engineering Centre, Charleroi, Belgium.
   index: 1
 - name: Department of Chemistry, University of Crete, Heraklion, Greece
   index: 2
 - name: Institute of Condensed Matter and Nanosciences, Université catholique de Louvain, B-1348 Louvain-la-Neuve, Belgium
   index: 3
 - name: Imperial
   index: 4
 - name: Toyota
   index: 5

date: January 2026
bibliography: paper.bib
---

# Summary



![Wetting-angle-kit logo](wetting-angle-kit_logo.pdf){width=50%}



Wetting-angle-kit is a Python toolkit that extracts wettability properties, such as the contact angle of a droplet on a surface, from molecular dynamics simulations. It is compatible with a variety of file formats (xyz, LAMMPS, ASE) and provides two measurement methods for contact angle analysis. Additionally, the package includes utilities for statistical processing and data visualization, making it a comprehensive solution for wettability studies.

# Statement of need

The measurement of contact angles in molecular dynamics simulations has evolved significantly since 1997, with key contributions in 2012, 2016, and 2024 [@Hautman1997; @Rafiee2012; @Vega2016; @Recent2024]. Despite this progress, the field lacks a standardized platform for comparing and validating the diverse methods used to measure contact angles. This fragmentation can hinder reproducibility and collaboration. Wetting-angle-kit aims to resolve this issue by offering a flexible, open-source toolkit where researchers can implement new methods, compare them with established techniques, and ensure a consistent baseline for molecular dynamics studies.







# Software Description, Features, and Computational Workflow



The code is organized into three modules that follow on from one another and are interdependent: the parser, the contact angle method and the visualization and statistics. In the following section, we describe the main principles of each module.

## The parser



The Parser Module is a core component of the wetting-angle-kit package, designed to handle and process molecular dynamics (MD) trajectory files from a variety of formats, including ASE-readable trajectory files, LAMMPS dump files, and extended XYZ files. By providing a unified interface, the module ensures consistent extraction of essential information (atomic coordinates, frame counts, and simulation box dimensions) from diverse trajectory formats and atomic structures.

At its core, the module relies on the BaseParser abstract base class (ABC), which defines the fundamental methods that all parsers must implement. These methods include:

Counting frames in a trajectory.

Parsing Cartesian coordinates for selected atoms.

Optionally retrieving simulation box dimensions.

By enforcing this standardized structure, the ABC guarantees that all trajectory and structure data is organized uniformly across parsers. This consistency not only ensures reliable data processing but also facilitates seamless integration with downstream contact angle measurement methods.

The module’s design prioritizes modularity and extensibility, enabling researchers to effortlessly incorporate support for new file formats or analysis techniques. By standardizing the parsing interface, the Parser module enables the wetting-angle-kit package to process various MD datasets consistently, thereby promoting reproducible and robust wettability analyses across different simulation configurations.





## The contact angle methods

The contact angle methods section provide two complementary approaches for estimating contact angles from molecular dynamics (MD) simulations: the slicing method and the binning method. Both methods are designed to analyze the geometry of liquid droplets on surfaces, but they employ distinct computational strategies to achieve robust and reproducible results. They inherit from the BaseContactAngleAnalyzer abstract base class (ABC), which defines a standardized interface for contact angle analysis. This design ensures that each method adheres to a consistent structure, providing a robust and extensible framework for researchers.

### Slicing method

The slicing method focuses on a analysis of each frame in the trajectory. By sampling radial lines from the droplet's geometric center, it fits circles to the liquid-vapor interface for each slice or inclination, depending on the chosen droplet geometry (cylindrical or spherical). This approach allows for precise determination of the contact angle by examining the intersection of the fitted circle with the substrate surface. The slicing method is particularly suited for the analysis of long trajectory files where the user needs to understand when the droplet reaches an equilibrium regime.

### Binning method

The binning method adopts a global approach by aggregating particle coordinates across multiple frames into a 2D spatial grid. This grid forms a time-averaged density field, which is then fitted using a hyperbolic tangent model to describe the liquid-vapor interface. The fitted model provides a smooth representation of the droplet geometry, from which the contact angle is derived. The binning method is ideal for symmetric droplets and scenarios where a global, averaged representation of the droplet is sufficient. Its strength lies in its ability to handle large datasets efficiently by reducing the dimensionality of the problem.



Together, these methods offer flexibility and precision. The package offers two complementary methods for contact angle analysis, each with distinct advantages and trade-offs. The SlicedContactAngleAnalyzer is ideal for high-precision, frame-by-frame analysis, particularly for complex or asymmetric droplets. However, its computational expense may not be suitable for large-scale simulations.

The BinnedContactAngleAnalyzer, on the other hand, is fast and efficient, making it well-suited for symmetric droplets and large datasets. However, it requires a sufficiently large sample size and may lack precision for irregular geometries.

These methods enable researchers to select the most suitable approach based on their specific requirements, considering trade-offs between precision, computational efficiency, and system complexity.

## The visualization modules

The package includes a comprehensive visualization and statistics module designed to facilitate the analysis and interpretation of contact angle measurements from molecular dynamics (MD) simulations. This module provides tools for statistical analysis, visual representation, and comparative studies, enabling researchers to derive meaningful insights from their simulation data.

The BaseTrajectoryAnalyzer abstract base class serves as the foundation for trajectory analysis, defining methods for computing statistics, generating visualizations, and extracting contact angles and surface areas. Derived classes, such as BinningTrajectoryAnalyzer and SlicedTrajectoryAnalyzer, implement these methods for specific analysis techniques.

For visualization, the module includes classes like DropletSlicedPlotter and DropletSlicedPlotterPlotly, which generate static and interactive plots of droplet slices, respectively. These tools allow users to visualize surface contours, fitted circles, and tangent lines, enhancing the interpretability of contact angle measurements. Additionally, the ContactAngleAnimator class generates interactive animations of contact angles per frame, providing a dynamic view of droplet behavior over the simulation timeline.

The MethodComparison utility enables comparative analysis across multiple trajectory analyzers, offering functions to overlay and juxtapose statistical results. This feature is particularly useful for validating results across different methods or simulation setups.

Overall, the visualization and statistics module add tools to analyze, visualize, and compare contact angle data, fostering a deeper understanding of wettability phenomena in MD simulations.



# Examples and Applications







# Acknowledgements

MSCA fellowship ..



Computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the FRS-FNRS under Grant No. 2.5020.11 and by the Walloon Region.



# References
