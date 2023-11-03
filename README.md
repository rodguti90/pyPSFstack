# pyPSFstack

[![DOI](https://zenodo.org/badge/449654051.svg)](https://zenodo.org/doi/10.5281/zenodo.10069030)

© Rodrigo Gutiérrez Cuevas

Python library for the numerical modelling of PSF stacks used to 
characterize single molecule fluorescence microscope.

## What is it?

`pyPSFstack` provides two packages that are designed to work together.
The first one is `pyPSFstack` which can be used to model stacks
of PSFs containing both phase and polarization diversities, and both
scalar and birefringent pupils can be used to shape the PSF, and 
by default a dipolar source next to an interface is assumed.  
The second is `torchPSFstack` which was built to mirror `pyPSFstack`
but using the neural network framework `PyTorch`. Its main use is
to retrieve scalar or birefringent pupils from a modeled or 
measured PSF stack using Zernike or pixel based models. 
Both packages have been written assuming the most precise model for the 
source, that is, a dipole close to an interface. It can model both 
fully polarized or unpolarized dipoles, and it also provides
accurate models for the three-dimensional blurring due to the 
non-negligible size of the emitters.

## Citing the code

If the code was helpful for your work, please consider citing it along with the accompanying paper.

## Contributions 

This code is written and maintained by Rodrigo Gutiérrez-Cuevas.

## How does it work?

The following detailed examples show how both packages work:
1. `example_pyPSFstack.ipynb` shows how `pyPSFstack` can be used to model 
stacks of PSFs with varying typed of pupils, diversities and blurring.
2. `example_torchPSFstack.ipynb` uses `pyPSFstack` to model stack of PSFs
which are then fed to `torchPSFstack` to test the retrieval of both 
scalar and birefringent pupils with Zernike and pixel-based models.
3. `example_exp_retrieval.ipynb` shows the retrieval of a birefringent 
pupil using real data. 
