"""Module containing the definition for all pupil diversities. """
import numpy as np
from ..pupil import PupilDiversity

class NoDiversity():
    """Class defining the absence of diversity."""
    def _forward(self, input):
        return input

class ZDiversity(PupilDiversity):
    """PupilDiversity subclass defining the phase diversity given by defocus.
    
    Attributes
    ----------
    z_list : list or ndarray
        List containing the defocus values used for the phase diversity.
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
        z_list : list or ndarray
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
        PupilDiversity.__init__(self, aperture_size, computation_size, N_pts)
        
        self.z_list = np.reshape(np.array(z_list), (1,1,-1))
        self.N_zdiv = len(z_list)
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).astype(np.cfloat)
        return np.exp(1j*2*np.pi*self.nf*self.z_list*(1-ur**2)**(1/2))
    

class DDiversity(PupilDiversity):
    """PupilDiversity subclass defining the phase diversity for z0 N defocus for blurring.
    
    Attributes
    ----------
    diff_del_list : list or ndarray
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
        diff_del_list : list or ndarray
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
        PupilDiversity.__init__(self, aperture_size, computation_size, N_pts)
        
        self.diff_del_list = np.reshape(np.array(diff_del_list), (1,1,-1))
        self.N_ddiv = len(diff_del_list)
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).astype(np.cfloat)
        return np.exp(1j*2*np.pi*self.diff_del_list*self.ni
                      *(1-(self.nf*ur/self.ni)**2)**(1/2)) 

class DerivativeDiversity(PupilDiversity):
    """PupilDiversity subclass defining the scalar diversity for derivative with respect to z0.
    
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
        PupilDiversity.__init__(self, aperture_size, computation_size, N_pts)
        
        self.m = m
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        nr, nc = ur.shape
        ur = ur.astype(np.cfloat)
        pupil_array = np.empty((nr,nc,self.m+1), dtype=np.cfloat)
        for l in range(self.m+1):
            pupil_array[...,l] = (1j*2*np.pi*self.ni*(1-(self.nf*ur/self.ni)**2)**(1/2))**l
        
        return pupil_array