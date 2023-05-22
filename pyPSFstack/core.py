"""Module containing the PSFStack class used to put everything together."""
import numpy as np
from scipy.special import jv 
from .functions import trim_stack
from .blurring.blurring import NoBlurring
# from .pupils.windows import NoPupil

class PSFStack():
    """Class used for computing stacks of PSF.
    
    Attributes
    ----------
    pupils : List of Pupil objects
        List of sources, windows and aberrations used to model the stack.
    N_pupils : int
        Number of pupils being used
    N_pts : int
        Number of points used for the computation.
    zdiversiy : ZDiversity object
        Object defining the defocus phase diversity.
    pdiversity : PDiversity object
        Object defining the polarization diversity.
    blurring : Blurring object
        Object defining the blurring model.

    Methods
    -------
    compute_psf_stack(orientation, N_trim)
        Returns the modeled PSF stack.
    model_experimenta_stack(bckgd_photons,N_photons,norm,bleach_amplitudes,N_pts,noise)
        Returns the PSF stack modelling an experimental situation such as noise, 
        background illumination, and photobleaching.
    """
    def __init__(self, 
                 pupils, 
                 zdiversity=None, 
                 pdiversity=None, 
                 blurring=NoBlurring()):
        """"Constructor.

        Parameters
        ----------
        pupils : List of Pupil objects
            List of sources, windows and aberrations used to model the stack.
        zdiversiy : ZDiversity object
            Object defining the defocus phase diversity.
        pdiversity : PDiversity object
            Object defining the polarization diversity.
        blurring : Blurring object
            Object defining the blurring model.
        psf_stack : ndarray
            Array containing the computed PSF stack.
        """
        self.pupils = pupils
        self.N_pupils = len(self.pupils)
        self.N_pts = self.pupils[0].N_pts
        self.zdiversity = zdiversity
        self.pdiversity = pdiversity
        self.blurring = blurring

    def compute_psf_stack(self, orientation=[0,0,0], N_trim=None):
        """Computes the modeled PSF stac which are stored in the attribute psf_stack.
        
        Parameters
        ----------
        orientation : list or ndarray
            List of length 3 representing the orientaiton for the dipole. 
            The special value [0,0,0] represents an unpolarized dipole while any
            other value will be taken as fully polarized. 
        N_trim : int, optional
            Whether to trim the resulting PSFs to a square arrys of size N_trim.
        """
        output = self._compute_compound_pupils()
        # diversities can be added to pupil sequence
        
        output = self.blurring.diversity._forward(output)

        if self.zdiversity is not None:
            output = self.zdiversity._forward(output)

        output = self._propagate_image_plane(output)

        if self.pdiversity is not None:
            output = self.pdiversity._forward(output)

        output = self.blurring.compute_blurred_psfs(output, orientation)
        
        if N_trim is not None:
            self.psf_stack = trim_stack(output, N_trim)
        else:
            self.psf_stack = output

    def _compute_compound_pupils(self):
        # output = self.pupils[0].get_pupil_array()
        output = self.pupils[0]._forward()
        for ind in range(self.N_pupils-1):
            output = self.pupils[ind+1]._forward(output)
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


    def model_experimental_stack(self, 
                                 bckgd_photons=20, 
                                 N_photons=200, 
                                 norm='average',
                                 bleach_amplitudes=1, 
                                 N_pts=None,
                                 noise=True):
        """Returns the PSF stack modelling an experimental situation.
        
        Parameters
        ----------
        bckgd_photons : int
            The number of background photons to be added to each pixel.
        N_photons : int
            Defines the number of photons in the PSFs according to norm.
        norm : {'average', 'max'}, optional
            'average' means that on average each the PSF for each diversity will
            have N_photons while 'max' means the the pixels with the highest intensity
            for all diversities has detects N_photons.
        bleach_amplitudes : ndarray
            Amplitudes changing the scale of the PSFs as a function of diversity.
        N_pts : int
            Whether to trim the resulting PSFs to a square arrys of size N_pts.
        noise : bool, optional
            Whether to add Poisson noise to the PSF stack.
            
        Returns
        -------
        stack : ndarray
            Array containing the PSF stack modelling an experimental situation.
        """
        rng = np.random.default_rng()
        
        if N_pts is None:
            N_pts = self.N_pts
        
        stack = trim_stack(self.psf_stack.copy(), N_pts)
       
        if norm=='average':
            ave = np.mean(np.sum(stack, axis=(0,1)))
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

    

    