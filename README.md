# Variational Kinetic Monte Carlo (VKMC)
This repository provides python code to use the Variational Principle for Mass Transport (Trinkle, phy. rev. lett. 2018) to compute transport coefficients in random alloys. This involves predicting a vector-valued function of atomic configurations, called the relaxation vector from single-step Kinetic Monte Carlo data. This vector is the solution to a Poisson equation in an abstract space of atomic configurations. We approximately solve for this vector using neural network and cluster expansion models, in combination with the Variational Principle, which provides a minimization problem to which they are the optimal solutions. We thus effectively try to predict transport coefficients from a single KMC step, reducing the computational load by millions of KMC calculations, making the method suitable for high-throughput type of applications with energy models that may be too expensive for standard Kinetic Monte Carlo methods.

All examples in this repository are mainly focused on FCC alloys, both model systems (non-interacting lattice gases) as well a realistic 5-component Cantor alloy (CoNiCrFeMn) simulated with a MEAM potential from Choi et. al. (npj Computational Materials 2018).

## Pre-requisites
 - Python 3.
 - Some Standard packages: numpy, scipy, numba, PyTorch(1.7 or higher), h5py.
 - Onsager package by Dallas. R. Trinkle, with the cluster modules (https://github.com/DallasTrinkle/Onsager/tree/cluster).
 - The ASE package (https://wiki.fysik.dtu.dk/ase/).
 - An installation of the LAMMPS package to use MEAM potentials and do climbing-image NEB calculations.
## Getting Started
## (1) Accessing the modules
The easiest way to access the required codes to perform transport coefficient calculations is to the add the "Symm_Network", "VCE" (short for Vector-valued Cluster Expansion) and "Utils" directories to the $PYTHONPATH variable or add them to your conda environment using the ```conda develop``` command.

## (2) Crystal Data File.
Nearly every calculation using the codes in this repository requires a hdf5 data file containing information about the crystal structure and supercells. The directory "CrysDat_FCC" contains example notebooks that show how to generate this required information about the FCC crystal structure. We have used 8x8x8 primitive FCC supercells with 512 sites. The notebook "Supercell_data_generation_1nn.ipynb" contains code to generate this data, while the notebook "Test_Crystal_Data_FCC.ipynb" has code to test the generated crystal data. An example hdf5 crystal data file "CrysData.h5" is also provided. Along with these, we have also used 5x5x5 orthogonal FCC supercells with 500 sites. The crystal data for this supercell is stored in the file "CrysData_ortho_5_cube.h5", and its generation and testing codes are also provided.

## (3) Examples
The "Examples" directory contains more jupyter notebooks as well as example job submission scripts for slurm to show to use the codes in this repository to perform transport coefficient calculations similar to the ones in our publication (in progress as of May 26, 2023). These examples directories are described as follows:

### (3.1) Lattice Gas Data Generation Example
The directory ```Examples/1_Binary_Lattice_Gas_data_Generation/``` contains the example notebook ```1_makeDataSet.ipynb```. This notebook shows how to use the ```Utils/LatGas.py``` module to generate single KMC-step data for a non-interacting, binary, FCC lattice gas which has fixed rates for vacancy atom-exchange (1.0 for the "fast" species, and 0.001 for the "slow" species). The notebook also illustrates the format in which the sinlge KMC step data must be saved so that it can be used by our neural network and cluster expansion codes.

### (3.2) Neural Network Example.
For predicting transport coefficients with machine learning, we predict the afore-mentioned relaxation vectors with Group Equivariant Convolutional Neural Networks (Cohen and Welling, 2016). The directory ```Examples/2_Neural_network_training/``` contains an example job script to launch neural network calculations using the module ```Symm_Network/GCNetRun.py``` for a variety of cases such as training and evaluating neural networks and computing relaxation vectors. The "ReadMe" file in the same directory contains more information about the command line arguments required during neural network runs.

### (3.3) Cluster Expansion Example.
The directory ```Examples/3_Cluster_expansion_training/``` contains an example job script to compute transport coefficients with linear basis cluster expansion models to approximate the relaxation vectors, using the module ```VCE/LBAM_dataset.py```. The "ReadMe" file in the same directory provides information about the command line arguments. This module makes use of the "cluster.py" and "supercell.py" modules in the Onsager package (see pre-requisites).

### (3.3) MEAM Potential Monte Carlo example.
The directory ```Examples/4_Monte_Carlo_LAMMPS/``` contains a ReadMe file and an example job script that show how to perform Monte Carlo simulations with LAMMPS for the CoNiCrFeMn 5-component FCC alloys. This is done using the python script ```Utils/MEAM_MC/Init_state_MC.py```. An example notebook is provided that shows how to save these generated atomic configuration into numpy arrays. These states are then to be used as initial states for KMC random walk calculations. 

### (3.4) MEAM Potential Kinetic Monte Carlo example.
The directory ```Examples/5_Kinetic_Monte_Carlo_LAMMPS/``` contains a ReadMe file and an example job script that show how to perform Kinetic Monte Carlo simulations with LAMMPS using climbing-image NEB calculations in the CoNiCrFeMn 5-component FCC alloys. This is done using the python script ```Utils/MEAM_MC/NEB_steps_multiTraj.py```.

## References
* Dallas R. Trinkle, "Variational principle for mass transport.", Physcial Review Letters (2018) [doi:doi.org/10.1103/PhysRevLett.121.235901]
* Won-Mi Choi, Yong Hee Jo, Seok Su Sohn, Sunghak Lee & Byeong-Joo Lee - "Understanding the physical metallurgy of the CoCrFeMnNi high-entropy alloy: an atomistic simulation study", npj Computational Materials, 2018 [doi: doi.org/10.1038/s41524-017-0060-9]. This MEAM potential was obtained from this paper.
* Taco Cohen and Max Welling - "Group equivariant convolutional networks", PMLR 48:2990-2999, 2016. This paper is the basis for the symmetry-conforming neural network design.

License - MIT License: https://opensource.org/license/mit
