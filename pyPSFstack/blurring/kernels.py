"""Module defining the kernel classes used for the blurring."""
import numpy as np
from scipy.special import jv, j1, spherical_jn
from math import factorial
from ..pupil import BlurringKernel

class BKSphere(BlurringKernel):
    """BlurringKernel subclass used to define the kernels for exact blurring 
    given a spherical emission by the bead.
    
    Parameters
    ----------
    diff_del_list : list or ndarray
        List of slices to use for the computation of the z integral
        for the exact blurring model.
    radius : float
        Radius of the fluorescent bead. 
    nf : float
        Index of refraction for the immersion medium of the microscope objective.
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, 
                 diff_del_list,
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):
        """Constructor.
        
        Parameters
        ----------
        diff_del_list : list or ndarray
            List of slices to use for the computation of the z integral
            for the exact blurring model.
        radius : float
            Radius of the fluorescent bead. 
        nf : float
            Index of refraction for the immersion medium of the microscope objective.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        BlurringKernel.__init__(self, aperture_size, computation_size, N_pts)
        self.diff_del_list = diff_del_list
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        # ur = ur[...,None]   
        rad_d = self.nf * (self.radius**2 - self.diff_del_list**2)**(1/2)
        bk = rad_d * j1(2*np.pi*ur[...,None]*rad_d) / ur[...,None]
        origin = ur == 0
        bk[origin,:] = rad_d**2 * np.pi

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap


class BKSASphere(BlurringKernel):
    """BlurringKernel subclass used to define the kernels for semi-analytic 
    blurring for a spherical emission by the bead.
    
    Parameters
    ----------
    radius : float
        Radius of the fluorescent bead. 
    nf : float
        Index of refraction for the immersion medium of the microscope objective.
    l_max : int
        l_max == m//2 defines the number of terms used in teh expansion.
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, 
                 m,
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):
        """Constructor.
        
        Parameters
        ----------
        m : int
            Integer identifying the order to use for the semianalyticl method. 
            m=0 produces a 2D blurring based on a convolution.
        radius : float
            Radius of the fluorescent bead. 
        nf : float
            Index of refraction for the immersion medium of the microscope objective.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        BlurringKernel.__init__(self, aperture_size, computation_size, N_pts)
        self.l_max = m//2
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        nr, nc = ur.shape
        ur = ur.astype(np.cfloat)
        bk = np.empty((nr,nc,self.l_max+1), dtype=np.cfloat) 
        
        origin = ur == 0
        for l in range(self.l_max+1):
            pref = (self.nf*self.radius)**(l+2) / (2**l * factorial(l) *self.nf**(2*l+1))
            bessel_term = spherical_jn(l+1, 2*np.pi*self.nf*self.radius*ur)/(2*np.pi*ur)**(l+1)
            bessel_term[origin] = 2**l * factorial(l) * self.nf * self.radius / factorial(2*l)
            bk[...,l] = pref * bessel_term

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap

