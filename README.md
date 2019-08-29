even_handed
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/even_handed.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/even_handed)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/even_handed/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/even_handed/branch/master)

Implementation of the even-handed subsystem selection for projection-based embedding.

J. Chem. Phys. 149, 144101 (2018); https://doi.org/10.1063/1.5050533


## Basic Procedure:
 - Run original projection-based embedding calculation using atoms card
   - Print out relevant info (overlap and localized molecular orbitals) using MATROP
   - Save the localized molecular orbitals (LMOs) of the entire system using the wfu file
 - Run the even-handed script to determine the correct LMOs to include
   - The script can be run by `python main.py -d path_to_folder -n output_filename`
 - Restart the embedding calculation from the wfu file
   - Run a single iteration DFT calculation on the LMOs from the wfu file.
   This sets a bunch of global variables in Molpro which are needed for embedding to work properly
   - Call the embed command with orbs correctly specified


## Caution:
 - These rules must be followed otherwise the even-handed script will not work!
 - Case 1: System size stays constant (e.g. SN2 reaction)
   - The atoms of each molecule in the reaction coordinate need to be in the same order in the xyz file
 - Case 2: Subsystem A size changes
   - Use the `--b` option with `main.py` so subsystem B is even-handedly selected instead of subsystem A
   - Make sure that all subsystem A (active) atoms come first in the xyz file
   - Make sure all atoms in subsystem B are in the same order between the product and reactant in their respective xyz files


## Sanity Check:
 - Plot the densities of the even-handed selection of the subsystems as a visual sanity check
 - Plot the overlap metrics to ideally see a large separation between the chosen orbitals (an overlap metric approaching 1)
 and the excluded orbitals (an overlap metric approaching 0)


## Examples:
 - There is a simple example for an SN2 reaction within `data` that outlines the basic procedure

### Copyright

Copyright (c) 2019, Sebastian Lee


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.0.
