"""
torchPupils
===========

This package contains the definitions of classes for various types of pupils,
separated into three modules:

aberrations
    Pupils commonly used to describe aberrations.
sources
    Pupils used to describe sources.
windows 
    Pupils defining windows used for shaping PSFs.
"""
from . import aberrations
from . import sources
from . import windows