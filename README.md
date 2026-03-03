# 2D-GBDM-CC: Gaussian Beam Diffraction Migration with Cross-Constraint

This repository provides the C source code and a test dataset (Sigsbee 2A) for the Gaussian Beam Diffraction Migration (GBDM) method using a dip-domain cross-constraint strategy.

## 1. Prerequisites
- **Compiler**: GCC/icc 
- **Parallelization**: MPI (OpenMPI)
- **Seismic Software**: [Seismic Unix (SU)](https://github.com/JohnWStockwellJr/SeisUnix) must be installed and added to your system PATH.

## 2. Compilation
Compile the program using the provided Makefile:make
The executable oper will be generated.

## 3. Workflow and Usage
The Sigsbee 2A model data is included for testing. The imaging workflow consists of the following steps:
# Step 1: Dip Field Estimation
Calculate the local dip field of the subsurface. This code assumes the dip field is pre-calculated using the plane-wave destruction (PWD) method [(We use the local slope estimation algorithm by Wang et al., 2022).](https://pubs.geoscienceworld.org/seg/geophysics/article/87/3/F1/612291/A-MATLAB-code-package-for-2D-3D-local-slope)
# Step 2: Initial Migration for Image Components
Run the program to perform the initial diffraction migration. This step generates the left and right imaging components (migleft.dat and migright.dat).
Bash
mpirun -np 30 ./oper
# Step 3: Envelope Calculation (using Seismic Unix)
Use the suenv tool from Seismic Unix to compute the envelope amplitudes of the imaging components. Execute the following commands in your terminal:
Bash
For the right component
suaddhead ns=1201 < migright.dat | suenv mode=amp | sustrip > mig_rightenv.dat
For the left component
suaddhead ns=1201 < migleft.dat | suenv mode=amp | sustrip > mig_leftenv.dat
Note: Ensure ns matches the number of samples in your depth axis.
# Step 4: Final Cross-Constraint Result
Run the program again. It will automatically load the previously generated components and their envelopes (mig_leftenv.dat and mig_rightenv.dat) to apply the cross-constraint imaging condition and output the final purified diffraction image.
## 4. Citation
If you use this code for your research, please cite our paper:
Pan, X., Yue, Y., et al., 2026. Gaussian Beam Diffraction Imaging via Dip-Domain Cross-Constraint Strategy. Computers & Geosciences.
## 5. License
This project is licensed under the MIT License - see the LICENSE file for details.
