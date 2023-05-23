"""Module containing the definition for diversity dependent misalignments.
"""
import torch
import torch.nn as nn
import numpy as np
from .pupil import torchScalarWindow


class torchDefocuses(torchScalarWindow):
    """Class for including diversity dependent defocuses to the optimization
    
    Attributes
    ----------
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    step_f : float
        Step size of at the BFP.
    sh : list
        Size of diversities.
    delta_zs : Tensor
        Tensor specifying defocuses.

    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, sh, aperture_size=1., computation_size=4., 
                 N_pts=128
                 ):
        """Constructor.

        Parameters
        ----------
        sh : list
            Size of diversities.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        super(torchDefocuses, self).__init__(aperture_size, computation_size, N_pts)
        
        self.delta_zs = nn.Parameter(torch.zeros(sh, requires_grad=True, dtype=torch.float))
        self.sh = list(sh)

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        npix = ur.shape[0]
        ur = torch.reshape(ur.type(torch.cfloat), [npix]*2 + [1]*len(self.sh))
        aperture = self.get_aperture(dummy_ind=len(self.sh))
        defocus = torch.exp(1j*2*np.pi*self.delta_zs*(1-ur**2)**(1/2))
        return aperture * defocus

class torchTilts(torchScalarWindow):
    """Class for including diversity dependent tilts to the optimization. 

    Attributes
    ----------
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    step_f : float
        Step size of at the BFP.
    sh : list
        Size of diversities.
    xt : Tensor
        Tensor specifying tilts along x.
    yt : Tensor
        Tensor specifying tilts along y.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, sh,
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        """Constructor.

        Parameters
        ----------
        sh : list
            Size of diversities.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        super(torchTilts, self).__init__(aperture_size, computation_size, N_pts)
        
        self.sh = list(sh)
        self.xt = nn.Parameter(torch.zeros(sh, requires_grad=True, dtype=torch.float))
        self.yt = nn.Parameter(torch.zeros(sh, requires_grad=True, dtype=torch.float))

    def get_pupil_array(self):

        ux, uy = self.xy_mesh()
        npix = ux.shape[0]
        ux = torch.reshape(ux, [npix]*2 + [1]*len(self.sh) )
        uy = torch.reshape(uy, [npix]*2 + [1]*len(self.sh) )
        return torch.exp(1j*2*torch.pi* (self.xt * ux + self.yt *uy))
    
