
import numpy as np
from scipy.special import jv 
from .functions import trim_stack
from .blurring import NoBlurring
# from .pupils.windows import NoPupil

class PSFStack():

    def __init__(self, 
                 pupils, 
                 zdiversity=None, 
                 pdiversity=None, 
                 blurring=NoBlurring()):

        self.pupils = pupils
        self.N_pupils = len(self.pupils)
        self.N_pts = self.pupils[0].N_pts
        self.zdiversity = zdiversity
        self.pdiversity = pdiversity
        self.blurring = blurring

    def set_ddiversity(self, ddiversity):
        self.ddiversity = ddiversity

    def compute_psf_stack(self, orientation=[0,0,0]):
        output = self._compute_compound_pupils()
        # diversities can be added to pupil sequence
        
        output = self.blurring.diversity.forward(output)

        if self.zdiversity is not None:
            output = self.zdiversity.forward(output)

        output = self._propagate_image_plane(output)

        if self.pdiversity is not None:
            output = self.pdiversity.forward(output)

        self.psf_stack = self.blurring.compute_blurred_psfs(output, orientation)


    def _compute_compound_pupils(self):
        output = self.pupils[0].get_pupil_array()
        for ind in range(self.N_pupils-1):
            output = self.pupils[ind+1].forward(output)
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

    # def _incoherent_sum(self, input):
    #     return np.sum(np.abs(input)**2, axis=(-2,-1))

    # def _coherent_dipole(self, input, vec):
    #     field = input @ (np.array(vec))
    #     return np.sum(np.abs(field)**2, axis=-1)



    def model_experimental_stack(self, 
                                 bckgd_photons=20, 
                                 N_photons=200, 
                                 norm='average',
                                 bleach_amplitudes=1, 
                                 N_pts=None,
                                 noise=True):
        rng = np.random.default_rng()
        
        if N_pts is None:
            N_pts = self.N_pts
        
        stack = trim_stack(self.psf_stack, N_pts)
       
        if norm=='average':
            ave = np.mean(np.sum(stack, axes=(0,1)))
            stack /= ave
        elif norm=='max':
            max_value = np.max(stack)
            stack /= max_value
        else:
            raise ValueError('Invalid option for norm')

        stack = np.round(N_photons*bleach_amplitudes*stack + bckgd_photons)
        if noise:
            stack = rng.poisson(stack)
        return stack

    

    