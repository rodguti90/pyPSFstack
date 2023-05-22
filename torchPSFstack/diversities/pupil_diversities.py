"""Module containing the definition for all torch pupil diversities. """
import torch as torch
import torch.nn as nn
from ..pupil import torchPupilDiversity


class torchNoDiversity(nn.Module):
    """Class defining the absence of diversity."""
    def forward(self, input):
        return input

class torchZDiversity(torchPupilDiversity):
    """torchPupilDiversity subclass defining the phase diversity given by defocus.
    
    Attributes
    ----------
    z_list : Tensor
        Tensor containing the defocus values used for the phase diversity.
    N_zdiv : int
        Number of phase diversity.
    nf : float
        Index of refraction for the immersion medium of the microscope objective.
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    step_f : float
        Step size of at the BFP.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, 
                 z_list, 
                 nf=1.518, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        """Constructor.

        Parameters
        ----------
        z_list : list, ndarray, or Tensor
            List containing the defocus values used for the phase diversity.
        nf : float
            Index of refraction for the immersion medium of the microscope objective.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        super(torchZDiversity, self).__init__(aperture_size, computation_size, N_pts)
        
        self.z_list = torch.reshape(torch.tensor(list(z_list)), (1,1,-1))
        self.N_zdiv = len(z_list)
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).type(torch.cfloat)
        zdiv = torch.exp(1j*2*torch.pi*self.nf*self.z_list*(1-ur**2)**(1/2))
        return zdiv.type(torch.cfloat)

class torchDerivativeDiversity(torchPupilDiversity):
    """torchPupilDiversity subclass defining the scalar diversity for derivative with respect to z0.
    
    Attributes
    ----------
    m : int
        Number of derivatives to compute.
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
    step_f : float
        Step size of at the BFP.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, 
                 m=2, 
                 ni=1.33, 
                 nf=1.518, 
                 aperture_size = 1., 
                 computation_size=4., 
                 N_pts=128):
        """Constructor.

        Parameters
        ----------
        m : int
            Number of derivatives to compute.
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
        super(torchDerivativeDiversity, self).__init__(aperture_size, computation_size, N_pts)
        
        self.m = m
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        nr, nc = ur.shape
        ur = ur.type(torch.cfloat)
        pupil_array = torch.empty((nr,nc,self.m+1), dtype=torch.cfloat)
        for l in range(self.m+1):
            pupil_array[...,l] = (1j*2*torch.pi*self.ni*(1-(self.nf*ur/self.ni)**2)**(1/2))**l
        
        return pupil_array


class torchDDiversity(torchPupilDiversity):
    """torchPupilDiversity subclass defining the phase diversity for z0 N defocus for blurring.
    
    Attributes
    ----------
    diff_del_list : list, ndarray, or Tensor
        List containing the distances to the interface used for the phase diversity.
    N_ddiv : int
        Number of phase diversities.
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
    step_f : float
        Step size of at the BFP.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, 
                 diff_del_list, 
                 ni=1.33, 
                 nf=1.518, 
                 aperture_size = 1., 
                 computation_size=4., 
                 N_pts=128):
        """Constructor.

        Parameters
        ----------
        diff_del_list : list, ndarray, or Tensor
            List containing the distances to the interface used for the phase diversity.
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
        super(torchZDiversity, self).__init__(aperture_size, computation_size, N_pts)
        
        self.diff_del_list = torch.reshape(torch.tensor(list(diff_del_list)), (1,1,-1))
        self.N_ddiv = len(diff_del_list)
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).type(torch.cfloat)
        return torch.exp(1j*2*torch.pi*self.diff_del_list*self.ni
                      *(1-(self.nf*ur/self.ni)**2)**(1/2)) 