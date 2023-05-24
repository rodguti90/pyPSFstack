"""Module defining the blurring classes."""
import numpy as np
from math import factorial
from ..diversities.pupil_diversities import NoDiversity, DDiversity, DerivativeDiversity
from ..blurring.kernels import BKSphere, BKSASphere
# from .core import PSFStack


class Blurring():
    """Blurring superclass.
    
    Attributes
    ----------
    diversity : PupilDiversity class
        Diversity that needs to be taken into account at the 
        BFP to comput blurring.

    Methods
    -------
    compute_blurred_psfs(input)
        Returns the blurred PSFs provided as input.
    """
    def __init__(self):
        """Constructor."""
        self.diversity = NoDiversity()  
    
    def compute_blurred_psfs(self, input):
        """Returns the blurred PSFs provided as input.

        Parameters
        ----------
        input : ndarray
            Array with a stack of PSFs

        Returns
        -------
        ndarray
            Array with the stack of blurred PSFs
        """
        raise NotImplementedError("Please Implement this method")

    def _compute_intensity(self, input, orientation):
        if orientation == [0,0,0]:
            return np.sum(np.abs(input)**2, axis=(-2,-1))
        else:
            field = input @ (np.array(orientation))
            return np.sum(np.abs(field)**2, axis=-1)

class NoBlurring(Blurring):
    """"Blurring subclass representing the absence of blurring."""
    def __init__(self):
        self.diversity = NoDiversity()

    def compute_blurred_psfs(self, input, orientation):
        return self._compute_intensity(input, orientation)


class ExactBlurring(Blurring):
    """"Blurring subclass representing the exact 3D blurring.
    
    Attributes
    ----------
    diversity : PupilDiversity class
        Diversity computing the PSFs at varying distance from th 
        interface in order to perform the integral over the volume 
        of the fluorescent bead.
    bk : BlurringKernel class
        Pupil object representing the blurring kernel to be used.

    Methods
    -------
    compute_blurred_psfs(input)
        Returns the blurred PSFs provided as input.
    """
    def __init__(self,
                 radius=0.,
                 diff_del_list=[],
                 emission='sphere',
                 ni=1.33,
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 ):
        """Constructor.
        
        Parameters
        ----------
        radius : float
            Radius of the fluorescent bead.
        diff_del_list : list or ndarray
            List of slices to use for the computation of the z integral
            for the exact blurring model.
        emission : {'sphere'}, optional
            Moddel to use for the emission. Only the sphere model has been
            implemented, shell model to come.  
        ni : float
            Index of refraction for the embedding medium of the source.
        nf : float
            Index of refraction for the immersion medium of the microscope objective.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        Blurring.__init__(self)
        
        self.diversity = DDiversity(diff_del_list, 
                                    ni=ni, 
                                    nf=nf, 
                                    aperture_size=aperture_size, 
                                    computation_size=computation_size, 
                                    N_pts=N_pts)

        if emission == 'sphere':
            self.bk = BKSphere(diff_del_list,
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts)
        

    def compute_blurred_psfs(self, input, orientation):
        output = self._compute_intensity(input, orientation)
        output = self.bk._forward(output)
        return np.abs(output)


class SABlurring(Blurring):
    """"Blurring subclass representing the semi-analytic model for bluring.

    The blurring subcalss provides an accurate and faster model for the 
    blurring due to the size of fluorescent beads and can be used to 
    define 2D and 3D models. 
    
    Attributes
    ----------
    diversity : PupilDiversity class
        Diversity computing the PSFs at varying distance from th 
        interface in order to perform the integral over the volume 
        of the fluorescent bead.
    m_max : int 
        Integer identifying the order to use for the semianalyticl method. 
        m_max=0 produces a 2D blurring based on a convolution. 
    bk : BlurringKernel class
        Pupil object representing the blurring kernel to be used.

    Methods
    -------
    compute_blurred_psfs(input)
        Returns the blurred PSFs provided as input.
    """
    def __init__(self,
                 radius=0.,
                 m=2,
                 emission='sphere',
                 ni=1.33,
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 ):

        Blurring.__init__(self)
        """Constructor.
        
        Parameters
        ----------
        radius : float
            Radius of the fluorescent bead.
        m : int 
            Integer identifying the order to use for the semianalyticl method. 
            m=0 produces a 2D blurring based on a convolution. 
        emission : {'sphere'}, optional
            Moddel to use for the emission. Only the sphere model has been
            implemented, shell model to come.  
        ni : float
            Index of refraction for the embedding medium of the source.
        nf : float
            Index of refraction for the immersion medium of the microscope objective.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        self.diversity = DerivativeDiversity(m=m, 
                                    ni=ni, 
                                    nf=nf, 
                                    aperture_size=aperture_size, 
                                    computation_size=computation_size, 
                                    N_pts=N_pts)
        self.m_max = m
        if emission == 'sphere':
            self.bk = BKSASphere(m,
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts)
        

    def compute_blurred_psfs(self, input, orientation):
        output = self._compute_even_intensity_der(input, orientation)
        output = self.bk.forward(output)
        return np.abs(output)
        # return output

    def _compute_even_intensity_der(self, input, orientation):
        
        if orientation == [0,0,0]:
            field_m = input
        else:
            field_m = input @ (np.array(orientation))
        
        in_sh = list(input.shape)
        l_max = self.m_max//2 +1
        int_m = np.zeros(in_sh[:2]+[l_max]+in_sh[3:], dtype=np.cfloat)

        for l in range(l_max):
            for k in range(2*l+1):
                int_m[:,:,l,...] += (factorial(2*l)/(factorial(k)*factorial(2*l-k))) * \
                    np.conj(field_m[:,:,k,...]) * field_m[:,:,2*l-k,...]
        
        if orientation == [0,0,0]:
            return np.sum(int_m, axis=(-2,-1))
        else:
            return np.sum(int_m, axis=-1)

