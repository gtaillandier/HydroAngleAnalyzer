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

Wetting-angle-kit is a Python toolkit designed to extract wettability properties, specifically the contact angle of a droplet on a surface, from molecular dynamics (MD) simulations.

It supports a variety of standard file formats including extended XYZ, LAMMPS, and ASE-readable trajectories and offers two distinct computational methods for contact angle analysis. Furthermore, the package includes robust utilities for statistical post-processing and data visualization, providing a comprehensive workflow for wettability studies.

# Statement of need

The measurement of contact angles in molecular dynamics simulations has advanced significantly since early methodologies were proposed in 1997, with notable developments occurring in 2012, 2016, and 2024 [1-4]. Despite these advancements, the field currently lacks a standardized, unified platform for comparing and validating the diverse methods used to derive contact angles. This fragmentation poses challenges to reproducibility and collaborative research.

Wetting-angle-kit addresses this gap by providing a flexible, open-source framework. It allows researchers to implement new post process of the MD simulation of contact angle, benchmark them against established techniques, and establish a consistent baseline for wettability analysis in molecular dynamics.

# Software Description, Features, and Computational Workflow

The software architecture is organized into three interdependent modules: the Parser, the Contact Angle Analyzer, and the Visualization and Statistics module. The following sections outline the core principles of each component.

## The parser

The Parser Module serves as the data collection layer of the package, designed to process MD trajectory files from various formats, including ASE-readable trajectories, LAMMPS dump files, and extended XYZ files.

Central to this module is the BaseParser abstract base class (ABC), which enforces a uniform interface for all file handlers. This architecture ensures the consistent extraction of critical simulation data, including:

Atomic coordinates for selected species.

Frame indices and counts.

Simulation box dimensions (periodic boundary conditions).

By strictly adhering to this standardized structure, the ABC guarantees that data is normalized across different input formats. This consistency facilitates seamless integration with downstream analysis methods and ensures extensibility, allowing researchers to easily incorporate support for additional file formats or simulation engines.

## The contact angle methods

This module provides two complementary computational approaches for estimating contact angles, both inheriting from the BaseContactAngleAnalyzer abstract base class. This design ensures that both methods adhere to a standardized interface while addressing different analytical needs.All methods must support the two main geometric models: **spherical** (for spherical cap droplets) and **cylindrical** (for filament-like droplets, analyzed along a specific axis)[Citation].

### Slicing method


![3D spherical droplet scheme](wetting_angle_kit_3d_droplet.png){width=50%}

![Sliced droplet scheme](wetting_angle_kit.png){width=50%}

The Slicing Method performs a discrete, frame-by-frame analysis of the trajectory. By sampling radial slices from the droplet's geometric center, the algorithm fits circles to the liquid-vapor interface for each inclination. This technique allows for the precise determination of the contact angle at the intersection of the fitted circle and the substrate.

The Slicing Method is particularly advantageous for analyzing temporal evolution in long trajectories, enabling users to identify when a droplet reaches an equilibrium regime. While it offers high information for complex trajectory files, it is computationally intensive.

### Binning method

The Binning Method utilizes a global averaging approach. It aggregates particle coordinates across multiple frames into a 2D spatial grid, generating a time-averaged density field. This density field is fitted with a hyperbolic tangent model to describe the liquid-vapor interface, from which the contact angle is derived.

This method is computationally efficient and ideal for symmetric droplets or scenarios where a global, averaged representation is preferred. It excels at handling large datasets by reducing the dimensionality of the problem, though it requires a sufficient sample size to generate smooth density profiles.

### Comparison

Together, these approaches offer a versatile toolkit:

SlicedContactAngleAnalyzer: Best for high-precision, temporal analysis of complex geometries.

BinnedContactAngleAnalyzer: Best for rapid, computationally efficient analysis of symmetric systems and large datasets.

## The visualization modules

The Visualization and Statistics module is designed to facilitate the interpretation of simulation results. Built upon the BaseTrajectoryAnalyzer ABC, this module defines standard methods for computing statistics, extracting surface areas of the droplets, and generating visual outputs.
Derived classes, such as BinningTrajectoryAnalyzer and SlicedTrajectoryAnalyzer, implement specific logic for their respective methods. Key visualization features include:  Static and Interactive Plotting: Classes such as DropletSlicedPlotter and DropletSlicedPlotterPlotly generate plots of droplet slices, visualizing surface contours, tangent lines, and fitted circles.

Animation: The ContactAngleAnimator class creates interactive animations of the contact angle evolution, offering a dynamic view of droplet behavior throughout the simulation.

Method Comparison: The MethodComparison utility allows users to overlay and juxtapose statistical results from different analyzers, essential for validating new methods against established baselines.



The BaseTrajectoryAnalyzer abstract base class serves as the foundation for trajectory analysis, defining methods for computing statistics, generating visualizations, and extracting contact angles and surface areas. Derived classes, such as BinningTrajectoryAnalyzer and SlicedTrajectoryAnalyzer, implement these methods for specific analysis techniques.

For visualization, the module includes classes like DropletSlicedPlotter and DropletSlicedPlotterPlotly, which generate static and interactive plots of droplet slices, respectively. These tools allow users to visualize surface contours, fitted circles, and tangent lines, enhancing the interpretability of contact angle measurements. Additionally, the ContactAngleAnimator class generates interactive animations of  contact angles per frame, providing a dynamic view of droplet behavior over the simulation timeline.

The MethodComparison utility enables comparative analysis across multiple trajectory analyzers, offering functions to overlay and juxtapose statistical results. This feature is particularly useful for validating results across different methods or simulation setups.

Overall, the visualization and statistics module add tools to analyze, visualize, and compare contact angle data, fostering a deeper understanding of wettability phenomena in MD simulations.



# Examples and Applications

To validate the capabilities of wetting-angle-kit, molecular dynamics simulations were conducted using LAMMPS. The study focused on the wetting behavior of water droplets on two distinct substrates: a multi-layer graphene sheet (representing graphite) and a crystalline polymer surface (approximating PTFE).

Simulation Setup : To ensure geometric consistency and isolate the effect of droplet size, the substrate atoms were fixed (frozen) to create a rigid, atomically flat surface. This simplification minimizes thermal fluctuations of the substrate, which is a common approximation in nanoscale wetting studies.

For each substrate, four independent simulations were performed with varying droplet sizes to assess size-dependence. The systems contained 500, 1000, 2000, and 6000 water molecules, respectively. Water interactions were modeled using the SPC/E potential [Citation], which has been identified in previous studies as highly suitable for wetting applications. Carbon-water interactions for the graphite surface were described using Lennard-Jones (LJ) potentials, while polymer interactions were derived from the OPLS-AA force field.

##Theoretical Framework: Modified Young’s Equation

To extract the macroscopic contact angle from nanoscale measurements, the relationship between the measured contact angle ($\theta$) and the droplet size is analyzed using the Modified Young’s Equation. This relationship accounts for line tension effects, which are significant at the nanoscale. The equation is linearized to facilitate extrapolation:

$$\cos\theta = \cos\theta_\infty - \frac{\tau}{\gamma_{LV}} \cdot \frac{1}{r_B}$$



By plotting $\cos\theta$ against the inverse of the contact radius (or an equivalent geometric parameter derived from contact area $A$), the data yields a linear trend. The slope of this line corresponds to the influence of line tension ($\tau$) and surface tension ($\gamma_{LV}$), while the intercept provides the contact angle of an infinite droplet ($\cos\theta_\infty$). This regression allows for the extrapolation of fundamental wettability properties from finite-sized nanodroplets.

##Validation Results
![Mean cos angle vs surface for graphite](menscosnalge_vs_surface_graphite.pdf){width=50%}

![Mean cos angle vs surface for PTFE](menscosnalge_vs_surface_ptfe.pdf){width=50%}
The analysis yielded a contact angle of 93° for the graphite surface. This result is consistent with literature values obtained using similar carbon-oxygen LJ parameters [Citation]. Similarly, the contact angles extracted for the PTFE surface (using OPLS-AA parameters) showed good agreement with expected interaction strengths. These results confirm the accuracy of the toolkit in reproducing standard wettability metrics.



# Acknowledgements

MSCA fellowship ..



Computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the FRS-FNRS under Grant No. 2.5020.11 and by the Walloon Region.



# References
