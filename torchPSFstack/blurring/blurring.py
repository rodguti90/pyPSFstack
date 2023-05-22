import torch
import torch.nn as nn
from math import factorial

from ..diversities.pupil_diversities import torchNoDiversity, torchDerivativeDiversity
from ..blurring.kernels import torchBKSASphere, torchBK2DSphere

class torchBlurring(nn.Module):
    """torchBlurring superclass.
    
    Attributes
    ----------
    diversity : torchPupilDiversity class
        Diversity that needs to be taken into account at the 
        BFP to comput blurring.

    Methods
    -------
    compute_blurred_psfs(input)
        Returns the blurred PSFs provided as input.
    """
    def __init__(self):
        """Constructor."""
        super(torchBlurring, self).__init__()
        
        self.diversity = torchNoDiversity()  

    def compute_blurred_psfs(self, input):
        """Returns the blurred PSFs provided as input.

        Parameters
        ----------
        input : Tensor
            Array with a stack of PSFs

        Returns
        -------
        Tensor
            Array with the stack of blurred PSFs
        """
        raise NotImplementedError("Please Implement this method")

    def _compute_intensity(self, input):
        return torch.sum(torch.abs(input)**2,dim=(-2,-1))

class torchNoBlurring(torchBlurring):
    """"torchBlurring subclass representing the absence of blurring."""
    def forward(self, input):
        return self._compute_intensity(input)

class torch2DBlurring(torchBlurring):
    """"torchBlurring subclass representing the 2D model for bluring.
    
    Attributes
    ----------
    bk : torchBlurringKernel class
        Pupil object representing the blurring kernel to be used.

    Methods
    -------
    compute_blurred_psfs(input)
        Returns the blurred PSFs provided as input.
    """
    def __init__(self,
                 radius=0.,
                 emission='sphere',
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 opt_radius=False
                 ):
        """Constructor.
        
        Parameters
        ----------
        radius : float
            Radius of the fluorescent bead.
        emission : {'sphere'}, optional
            Moddel to use for the emission. Only the sphere model has been
            implemented, shell model to come.  
        nf : float
            Index of refraction for the immersion medium of the microscope objective.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        opt_radius : bool, optional
            Whether to ass the radius paramter to the optimization.
        """
        super(torch2DBlurring, self).__init__()


        if emission == 'sphere':
            self.bk = torchBK2DSphere(
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts,
                               opt_radius=opt_radius)

    def forward(self, input):
        output = self._compute_intensity(input)
        output = self.bk(output)
        return torch.abs(output)


class torchSABlurring(torchBlurring):
    """"torchBlurring subclass representing the secon-order semi-analytic model for bluring.

    
    Attributes
    ----------
    diversity : torchPupilDiversity class
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
                 emission='sphere',
                 ni=1.33,
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 opt_radius=False
                 ):
        """Constructor.
        
        Parameters
        ----------
        radius : float
            Radius of the fluorescent bead.
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
        opt_radius : bool, optional
            Whether to ass the radius paramter to the optimization.
        """
        super(torchSABlurring, self).__init__()

        self.diversity = torchDerivativeDiversity(
                                    ni=ni, 
                                    nf=nf, 
                                    aperture_size=aperture_size, 
                                    computation_size=computation_size, 
                                    N_pts=N_pts)

        if emission == 'sphere':
            self.bk = torchBKSASphere(
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts,
                               opt_radius=opt_radius)

    def forward(self, input):
        output = self._compute_0N2_intensity_der(input)
        output = self.bk(output)
        return torch.abs(output)

    def _compute_0N2_intensity_der(self, input):
    
        field_m = input
        in_sh = list(input.shape)
        l_max = 2
        int_m = torch.zeros(in_sh[:2]+[2]+in_sh[3:], dtype=torch.cfloat)

        int_m[:,:,0,...] = torch.abs(field_m[:,:,0,...])**2
        int_m[:,:,1,...] = torch.conj(field_m[:,:,0,...]) * field_m[:,:,2,...] \
                            + 2 *  torch.abs(field_m[:,:,1,...])**2  \
                            + field_m[:,:,0,...] * torch.conj(field_m[:,:,2,...] )
        
        return torch.sum(int_m, dim=(-2,-1))
        