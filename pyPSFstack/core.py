
import numpy as np
from .functions import trim_stack
from .pupils.windows import NoPupil

class PSFStack():

    def __init__(self, pupils, zdiversity=None, pdiversity=None):
        self.pupils = pupils
        self.N_pupils = len(self.pupils)
        self.N_pts = self.pupils[0].N_pts
        self.zdiversity = zdiversity
        self.pdiversity = pdiversity
        # self.N_pdiv = pdiversity.N_pdiv
        # self.N_zdiv = zdiversity.N_zdiv

    def compute_psf_stack(self, orientation=[0,0,0]):
        output = self._compute_compound_pupils()

        if self.zdiversity is not None:
            output = self._compute_zdiv(output)

        output = self._propagate_image_plane(output)

        if self.pdiversity is not None:
            output = self._compute_pdiv(output)

        if orientation == [0,0,0]:
            self.psf_stack = self._incoherent_sum(output)
        else:
            self.psf_stack = self._coherent_dipole(output, orientation)

    def _compute_compound_pupils(self):
        output = self.pupils[0].get_pupil_array()
        for ind in range(self.N_pupils-1):
            output = self.pupils[ind+1].get_pupil_array() \
                @ output
        return output
        
    def _compute_zdiv(self, input):
        zdiv = self.zdiversity.get_pupil_array()
        output = input[...,np.newaxis,:,:] * zdiv[...,None,None]
        return output

    def _propagate_image_plane(self, input):
        # Note that this way of computing adds a linear phase at the image plane
        # which does not matter is we only care about the intensity distribution
        output = np.fft.fftshift(
            np.fft.fft2(input, 
                axes=(0,1),
                s=(self.N_pts,self.N_pts)), 
            axes=(0,1))/self.N_pts
        return output

    def _compute_pdiv(self, input):
        output = self.pdiversity.jones_list @ \
            input[...,np.newaxis,:,:]
        return output

    def _incoherent_sum(self, input):
        return np.sum(np.abs(input)**2, axis=(-2,-1))

    def _coherent_dipole(self, input, vec):
        field = input @ (np.array(vec))
        return np.sum(np.abs(field)**2, axis=-1)


    def model_experimental_stack(self, 
                                 bckgd_photons=20, 
                                 N_photons=200, 
                                 bleach_amplitudes=1, 
                                 N_pts=None):
        rng = np.random.default_rng()
        max_value = np.max(self.psf_stack)
        if N_pts is not None:
            stack = trim_stack(self.psf_stack, N_pts)/max_value
        else:
            stack = self.psf_stack/max_value
        stack = np.round(N_photons*bleach_amplitudes*stack + bckgd_photons)
        stack = rng.poisson(stack)
        return stack

    

    