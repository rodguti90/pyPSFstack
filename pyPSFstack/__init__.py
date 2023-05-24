"""
pyPSFstack
==========

Provides
  1. Pupils defining common sources, aberrations, and windows used for PSF shaping.
  2. Simple modelling of stacks of PSFs with phase and polarization diversities.
  3. Various blurring models for fluorescent beads

Available subpackages
---------------------
blurring
    Classes definiting various blurring models.
diversities
    Classes defining phase and polarization diversities.
pupils
    Classes defining pupils representing sources, aberrations and windows.
core
    Contains the main class pyPSFstack used to compute stacks of PSFs.
"""
from . import blurring
from . import diversities
from . import pupils
from .core import PSFStack