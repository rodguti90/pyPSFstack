"""
Created on Thu Dec 16 08:57:52 2021

@author: rodrigo
"""

import numpy as np
from functions import trim_stack
from pyPSFstack_old.pupils.windows import NoPupil

class PSFStack():

    def __init__(self, pupils=[NoPupil()], zdiversity=None, pdiversity=None):
        self.pupils = pupils
        self.N_pupils = len(self.pupils)
        self.N_pts = self.pupils[0].N_pts
        self.zdiversity = zdiversity
        self.pdiversity = pdiversity
        # self.N_pdiv = pdiversity.N_pdiv
        # self.N_zdiv = zdiversity.N_zdiv

    def compute_psf_stack(self):
        self._compute_compound_pupils()
        if self.zdiversity is not None:
            self._compute_zdiv()
        self._propagate_image_plane()
        if self.pdiversity is not None:
            self._compute_pdiv()

        self._incoherent_sum()

    def _compute_compound_pupils(self):
        self.compound_pupils = [self.pupils[0].pupil_array]
        for ind in range(self.N_pupils-1):
            self.compound_pupils += [self.pupils[ind+1].pupil_array 
                @ self.compound_pupils[ind]]
        
    def _compute_zdiv(self):
        self.zdiv_pupil = self.compound_pupils[-1][...,np.newaxis,:,:] \
            * self.zdiversity.pupil_array[...,np.newaxis,np.newaxis]

    def _propagate_image_plane(self):    
        self.zdiv_psf = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(self.zdiv_pupil, axes=(0,1)), 
                axes=(0,1)), 
            axes=(0,1))/self.N_pts

    def _compute_pdiv(self):
        self.field_psf_stack = self.pdiversity.jones_list @ self.zdiv_psf[...,np.newaxis,:,:]

    def _incoherent_sum(self):
        self.psf_stack = np.sum(np.abs(self.field_psf_stack)**2,axis=(-2,-1))


    def model_experimental_stack(self, bckgd_photons=20, N_photons=200, bleach_amplitudes=1, N_pts=None):
        rng = np.random.default_rng()
        max_value = np.max(self.psf_stack)
        if N_pts is not None:
            stack = trim_stack(self.psf_stack, N_pts)/max_value
        else:
            stack = self.psf_stack/max_value
        stack = np.round(N_photons*bleach_amplitudes*stack + bckgd_photons)
        stack = rng.poisson(stack)
        return stack

    

    


class Microscope():
    '''
    Defines all the parameters of the microscope used in the experiments.

    This is used to compute all the parameters recquired for the propagatror
    from those of the experimental system.
    '''
    def __init__(self, wavelength=525, distance_coverslip=100, nf=1.518, NA=1.49, magnification=100, cam_pixel_size = 6500):
        self.cam_pixel_size = cam_pixel_size


