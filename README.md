# 2D-GBDM-CC
2D Gaussian Beam Diffraction Migration with Cross-Constraint Strategy.

## Description
This package implements 2D Gaussian Beam Diffraction Migration (GBDM) using a dip-domain cross-constraint strategy, an enhanced .

## Dependencies
- GCC compiler
- MPI (OpenMPI/MPICH)
- SeismicUnix (for data I/O)

## Compilation
Use the provided Makefile:
```bash
make
mpirun -np [nodes] ./oper
