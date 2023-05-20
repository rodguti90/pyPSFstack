"""Module containing the definition of windows used for shaping PSFs. 
"""
import numpy as np
from ..pupil import BirefringentWindow, ScalarWindow
# from ..diversities.pola_diversities import jones_qwp

# class NoPupil(Pupil):
#     def __init__(sefl):
#         pass 
    
#     def get_pupil_array(self):
#         return np.array([[1,0],[0,1]])

class Defocus(ScalarWindow):
    """ScalarWindow subclass for representing the phase mask for defocus.

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
    nf : float
        Defines the index of refraction of the immersion medium used for the objective.
    delta_z : float
        Defines the defocus amount normalized by the wavelength, should be 
        negative if the objective is moving closer to the source. 
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, nf=1.518, delta_z=0
                 ):
        """Constructor.

        Parameters
        ----------
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        nf : float
            Defines the index of refraction of the immersion medium used for the objective.
        delta_z : float
            Defines the defocus amount normalized by the wavelength.
        """
        super(Defocus, self).__init__(aperture_size, computation_size, N_pts)
        
        self.nf = nf
        self.delta_z = delta_z
        
    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = ur.astype(np.cfloat)
        aperture = self.get_aperture(dummy_ind=0)
        defocus = np.exp(1j*2*np.pi*self.nf*self.delta_z*(1-ur**2)**(1/2))
        return aperture * defocus


class SEO(BirefringentWindow):
    """BirefringentWindow subclass for representing stress-engineered optics (SEO).

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
    c : float
        Defines the scaling parameter.
    phi : float
        Defines the orientation.
    center : list or ndarray
        Defines the center.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi, phi=0, center=[0,0]):
        """Constructor.
        
        Parameters
        ----------
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        c : float
            Defines the scaling parameter.
        phi : float
            Defines the orientation.
        center : list or ndarray
            Defines the center.
        """
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        self.c = c
        self.phi = phi
        self.center = center

    def get_pupil_array(self):
        ux, uy = self.xy_mesh()
        ny, nx = ux.shape
        uxt = ux - self.center[0]
        uyt = uy - self.center[1]
        ur = np.sqrt(uxt**2 + uyt**2)
        uphi = np.arctan2(uyt, uxt)
        jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
        jones_mat[...,0,0] = np.cos(self.c*ur/2) +1j*np.sin(self.c*ur/2)*np.cos(uphi-2*self.phi)
        jones_mat[...,0,1] = -1j*np.sin(self.c*ur/2)*np.sin(uphi-2*self.phi)
        jones_mat[...,1,0] = jones_mat[...,0,1]
        jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
        return self.get_aperture() * jones_mat


class Qplate(BirefringentWindow):
    """BirefringentWindow subclass for representing q-plates.

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
    q : float
        Defines the charge for the q_late.
    alpha : float
        Defines the orientation.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, q=1, alpha=0):
        """Constructor.
        
        Parameters
        ----------
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        c : float
            Defines the charge for the q_plate.
        alpha : float
            Defines the orientation.
        """
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        self.q = q
        self.alpha = alpha

    def get_pupil_array(self):
        _, uphi = self.polar_mesh()
        ny, nx = uphi.shape
        jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
        theta = self.q*uphi + self.alpha
        jones_mat[...,0,0] = 1j*np.cos(2*theta)
        jones_mat[...,0,1] = 1j*np.sin(2*theta)
        jones_mat[...,1,0] = -np.conj(jones_mat[...,0,1])
        jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
        return self.get_aperture() * jones_mat


# def jones_qplate(uphi, q, alpha):
#     ny, nx = uphi.shape
#     jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
#     theta = q*uphi + alpha
#     jones_mat[...,0,0] = 1j*np.cos(2*theta)
#     jones_mat[...,0,1] = 1j*np.sin(2*theta)
#     jones_mat[...,1,0] = -np.conj(jones_mat[...,0,1])
#     jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
#     return jones_mat

# def jones_seo(ux, uy, c=1.24*np.pi, phi=0, center=np.array([0,0])):
#     ny, nx = ux.shape
#     uxt = ux - center[0]
#     uyt = uy - center[1]
#     ur = np.sqrt(uxt**2 + uyt**2)
#     uphi = np.arctan2(uyt, uxt)
#     jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
#     jones_mat[...,0,0] = np.cos(c*ur/2) +1j*np.sin(c*ur/2)*np.cos(uphi-2*phi)
#     jones_mat[...,0,1] = -1j*np.sin(c*ur/2)*np.sin(uphi-2*phi)
#     jones_mat[...,1,0] = jones_mat[...,0,1]
#     jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
#     return jones_mat



