# SLPI-DM-2020

C++ implementation of various iterative linear solvers. We did not bother to create our own matrix class and used the Eigen matrices and vectors.

This code was an assignment, but it may be upgraded in the future.

## Main features
* Minimal Residual, Optimal Step Gradient, Conjugate Gradient (also available with a diagonal preconditionner)
* Krylov Methods : FOM, GMRes and their SPD versions (not really working yet)
* Read matrices from .mtx files (MatrixMarket)
* Nice I/Os thanks to the termcolor library.

## Features to be added in the future
- [ ] Implement new solvers : BiCG, BiCGStab... maybe others ?
- [ ] Implement a preconditionner class, and template the Solver class with it.
- [ ] Parallelize the code (not for now...).
