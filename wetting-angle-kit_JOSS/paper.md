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



# Statement of need

# Software Description, Features, and Computational Workflow

The code is organized into four modules that can be combined into a complete workflow or used independently. We describe here the overall working principles of each module. For a more practical approach, we provide [online tutorials](https://abinit.github.io/abipy_book/lumabi/lumiwork/lesson_lumiwork.html).


## LumiWork Module

![The LumiWork module, an AbiPy Workflow that automates ABINIT DFT tasks with $\Delta$SCF constrained occupations.](LumiWork.pdf)

A computational workflow for calculating phonon-resolved photoluminescence (PL) spectra of defect systems starts with the LumiWork module, which automates ABINIT DFT tasks with $\Delta$SCF constrained occupations [@jones1989density;@hellman2004potential]. Users provide the defect supercell structure, the DFT input parameters, and constrained occupations of the Kohn-Sham states designed to mimic the excited state of the system under study. This module manages two structural relaxations for
the ground- and the excited-state, and offers optional static SCF computations followed by non-SCF band structure calculations. As the relaxed excited state is not known in advance, input files are generated dynamically.

## $\Delta$SCF Post-Processing Module



## IFCs Embedding Module

## Lineshape Calculation Module



# Examples and Applications

This computational workflow has been used for inorganic phosphors activated with Eu$^{2+}$ dopants and has also been tested on a variety of other systems including  F-centers (oxygen vacancy) in CaO and the NV center in diamond. Its versatility allows for any kind of point defect.
These developments have been particularly useful in understanding the luminescence properties of technologically significant red-emitting Eu-doped phosphor materials. Notably, the workflow has been applied to SrAl$_2$Li$_2$O$_2$N$_2$:Eu$^{2+}$ and SrLiAl$_3$N$_4$:Eu$^{2+}$, shedding new light on their phonon sideband [@bouquiaux2021importance;@bouquiaux2023first]. We refer the reader to the accompanying notebook tutorials for practical examples demonstrating the application of this workflow.


# Acknowledgements
MSCA fellowship ..

Computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the FRS-FNRS under Grant No. 2.5020.11 and by the Walloon Region.

# References