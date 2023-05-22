"""Module containing the definitions for the sources.
"""
import numpy as np
from ..pupil import Source


class DipoleInterfaceSource(Source):
    """Pupil subclass used for defining the Green tensor of a dipolar source near an interface.
    
    DipoleInterfaceSource defines the Green tensor at the back focal plane 
    for a point dipolar source located near an interface between its embedding
    medium and the immersion medium for the objective. This expression takes
    into account the supercritical angle radiation and the Fresnel coefficients.

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
    ni : float
        Index of refraction for the embedding medium of the source.
    nf : float
        Index of refraction for the immersion medium of the microscope objective.
    delta : float
        Distance of the dipole to the interface difined as positive in units of wavelength.
    alpha : float
        Parameter defining the reference focal plane. 
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, ni=1.33, nf=1.518, delta=0.1, alpha=None):
        """Constructor.

        Parameters
        ----------
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        ni : float
            Index of refraction for the embedding medium of the source.
        nf : float
            Index of refraction for the immersion medium of the microscope objective.
        delta : float
            Distance of the dipole to the interface difined as positive in units of wavelength.
        alpha : float
            Parameter defining the reference focal plane. 
        """
        Source.__init__(self, aperture_size, computation_size, N_pts)
        
        self.ni = ni
        self.nf = nf
        self.delta = delta
        if alpha is None:
            nr = nf/ni
            self.alpha = (11*nr**3 + 32*nr**5)/(3 + 16*nr**2 + 24*nr**4)
        else:
            self.alpha = alpha
            
    def get_pupil_array(self):     

        ux, uy = self.xy_mesh()
        ur, _ = self.polar_mesh()
        saf_defocus = compute_SAF_defocus(ur.astype(np.cfloat), 
            self.ni, self.nf, self.delta, self.alpha) 
        green = compute_green(ux, uy, self.ni, self.nf)
        aperture = self.get_aperture()
        return aperture * saf_defocus * green      

def compute_SAF_defocus(ur, ni, nf, delta, alpha):
    """Defines the scalar terms containing the SAF and defocus.
    """
    N_pts = ur.shape[0]
    saf_defocus = np.empty((N_pts,N_pts,1,1), dtype=np.cfloat)

    saf_defocus[...,0,0] =  np.exp(1j*2*np.pi*nf*delta
                        *((ni/nf)*(1-(nf*ur/ni)**2)**(1/2)
                        -alpha*(1-ur**2)**(1/2)))   
    return saf_defocus

def compute_green(ux, uy, ni, nf):
    """Defines the matrix components of the Green tensor at the BFP.
    """
    np.seterr(divide='ignore', invalid='ignore')
    
    ur2 = (ux**2 + uy**2).astype(np.cfloat)
    N_pts = ux.shape[0]

    # Compute the Phi coefficients which include the Fresnel coefs
    Phi1 = 2 * nf**2 * (1 - ur2)**(1/2) / \
        (nf * ni * (1 - nf**2 * ur2 / ni**2)**(1/2)+ ni**2 * (1 - ur2)**(1/2)) 
    Phi2 = (2  * nf * (1 - nf**2 * ur2 / ni**2)**(1/2)) / \
        (nf * (1 - nf**2 * ur2 /ni**2)**(1/2)+ ni * (1 - ur2)**(1/2))
    Phi3 = 2 * nf * (1 - ur2)**(1/2) / \
        (ni * (1 - nf**2 * ur2 / ni**2)**(1/2)+ nf * (1 - ur2)**(1/2))
    # The conservation of energy factor
    con_en = np.empty((N_pts, N_pts, 1, 1), dtype=np.cfloat)
    con_en[...,0,0] = (1 - ur2)**(1/4)

    green_mat = np.empty((N_pts, N_pts, 2, 3), dtype=np.cfloat)

    green_mat[...,0,0] = (ux**2 * (1- ur2)**(1/2) * Phi2 + uy**2 * Phi3)/ ur2  
    green_mat[...,0,1] = (ux * uy * ((1- ur2)**(1/2) * Phi2 - Phi3))/ ur2
    green_mat[...,0,2] = - ux * Phi1
    green_mat[...,1,1] = (uy**2 * (1- ur2)**(1/2) * Phi2 + ux**2 * Phi3)/ ur2
    green_mat[...,1,2] = - uy * Phi1
    green_mat[...,1,0] = green_mat[...,0,1]

    origin = ur2 == 0
    green_mat[origin,0,0] = Phi2[origin]
    green_mat[origin,1,1] = Phi3[origin]
    green_mat = green_mat / con_en
    green_mat = np.nan_to_num(green_mat, nan=0.0, posinf=0.0, neginf=0.0)

    return green_mat


# class DipoleSource(Pupil):
#     def __init__(self, aperture_size=1., computation_size=4., 
#                  N_pts=128, ni=1.33, nf=1.518, delta=0.1, alpha=None):
#         Pupil.__init__(self, aperture_size, computation_size, N_pts)
        
#         self.ni = ni
#         self.nf = nf
#         self.delta = delta
#         if alpha is None:
#             self.alpha = (self.nf/self.ni)**3
            
#     def get_pupil_array(self):     

#         ux, uy = self.xy_mesh()
#         ur, _ = self.polar_mesh()
#         saf_defocus = compute_SAF_defocus(ur.astype(np.cfloat), 
#             self.ni, self.nf, self.delta, self.alpha) 
#         green = compute_green(ux, uy, self.ni, self.nf)
#         aperture = self.get_aperture()
#         return aperture * saf_defocus * green 

# def fresnel_tp(ur, ni, nf):
#     (2  * nf * (1 - nf**2 * ur2 / ni**2)**(1/2)) / \
#         (nf * (1 - nf**2 * ur2 /ni**2)**(1/2)+ ni * (1 - ur2)**(1/2))