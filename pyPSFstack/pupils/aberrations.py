"""Module containing the definition of all windows used for aberrations. 
"""
import numpy as np
from ..pupil import BirefringentWindow, ScalarWindow
from .zernike_functions import zernike_sequence, defocus_j

class UnitaryZernike(BirefringentWindow):
    """BirefringentWindow subclass for representing Zernike-based aberrations.

    UnitaryZernike defines a pupil composed of a general Jones matrix where its elements
    are expanded in terms of Zernike polynomials. 
    Note: The piston and defocus terms of the scalar phase term are ommitted.

    Attributes
    ----------
    c_W : list or ndarray
        Zernike expansion coefficients for the phase term.
    c_q : list or ndarray
        Zernike expansion coefficients for the matrix term.
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    step_f : float
        Step size of at the BFP.
    index_convention : {'standard','fringe'}, optional
        Defines which single index convention to use for the Zernike polynomials.
    defocus_j : int
        Index specifying the defocus term.

    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, c_W, c_q, aperture_size=1., computation_size=4., 
                 N_pts=128, index_convention='standard'):
        """Constructor.
        
        Parameters
        ----------
        c_W : list or ndarray
            Zernike expansion coefficients for the phase term.
        c_q : list or ndarray
            Zernike expansion coefficients for the matrix term.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        index_convention : {'standard','fringe'}, optional
            Defines which single index convention to use for the Zernike polynomials.
        
        Methods
        -------
        get_pupil_array()
            Computes the pupil array.
        plot_pupil_field()
            Plots specified components of the array for the pupil.
        """
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        
        # assert len(jmax_list) == 5
        # self.jmax_list = jmax_list
        self.jmax_list = []
        for i in range(4):
            self.jmax_list += [len(c_q[i])]
        self.jmax_list += [len(c_W)+2]
        self.c_q = c_q
        self.c_W = c_W

        if index_convention not in ['standard','fringe']:
            raise ValueError('Invalid index convention.')
        self.index_convention = index_convention
        
        self.defocus_j = defocus_j(index_convention)

    def get_pupil_array(self): 

        x, y = self.xy_mesh()
        N_pts = x.shape[0]
        zernike_seq = zernike_sequence(np.max(self.jmax_list), 
                                        self.index_convention, 
                                        x/self.aperture_size, 
                                        y/self.aperture_size)
        # Computes the unitary decomposition matrix and scalar terms         
        #initialize the matrix term   
        Q = np.zeros((N_pts,N_pts,2,2), dtype=np.cfloat)
        qs = np.zeros((N_pts,N_pts,4))   
        #Compute the qs
        for k in range(4):
            qs[...,k] = np.sum(zernike_seq[...,:self.jmax_list[k]] 
                * self.c_q[k], axis=2)        
        #Compute the matrix term    
        Q[...,0,0] = qs[...,0] + 1j * qs[...,3]
        Q[...,0,1] = qs[...,2] + 1j * qs[...,1]
        Q[...,1,0] = -np.conj(Q[...,0,1])
        Q[...,1,1] = np.conj(Q[...,0,0])
        #Compute the wavefront
        # the phase term needs to be computed separetly since it
        # omits specific zernikes (piston 0 and defocus 4)       
        temp = np.hstack((self.c_W[:self.defocus_j-1], [0],
             self.c_W[self.defocus_j-1:]))     
        W = np.sum(zernike_seq[...,1:self.jmax_list[4]] * temp, axis = 2)
        
        #Compute the scalar term
        Gamma = np.empty((N_pts,N_pts,1,1), dtype=np.cfloat)
        Gamma[...,0,0] = np.exp(1j * 2 * np.pi * W)
        
        return Gamma * Q        

class UnitaryPixels(BirefringentWindow):
    """BirefringentWindow subclass for representing pixel-based aberrations.

    UnitaryPixels defines a pupil composed of a general Jones matrix where its elements
    are expanded in terms of the pixels defined by their discretization. 

    Attributes
    ----------
    W : ndarray
        Array representing the scalar aberrations
    qs : ndarray
        Array containing the components for the polarization aberrations.
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
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, W, qs, aperture_size=1., computation_size=4., 
                 N_pts=128):
        """Constructor.
        
        Parameters
        ----------
        W : ndarray
            Array representing the scalar aberrations
        qs : ndarray
            Array containing the components for the polarization aberrations.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        
        self.W = W
        self.qs = qs

    def get_pupil_array(self): 
        
        aperture = self.get_aperture(dummy_ind=0)
        N_pupil = aperture.shape[0]

        Q = np.zeros((N_pupil,N_pupil,2,2), dtype=np.cfloat)
        Q[...,0,0] = self.qs[0] + 1j*self.qs[3]
        Q[...,0,1] = self.qs[2] + 1j*self.qs[1]
        Q[...,1,0] = -self.qs[2] + 1j*self.qs[1]
        Q[...,1,1] = self.qs[0] - 1j*self.qs[3]

        Gamma = aperture*np.exp(1j*2*np.pi*self.W)

        return (Gamma[...,None,None] * Q)
   

class ScalarZernike(ScalarWindow):
    """ScalarWindow subclass for representing Zernike-based scalar aberrations.

    ScalarZernike defines a scalar pupil composed of amplitude and phase terms which
    are expanded in terms of Zernike polynomials. 
    Note: The piston and defocus termsfor the phase term are ommitted.

    Attributes
    ----------
    c_A : list or ndarray
        Zernike expansion coefficients for the matrix term.
    c_W : list or ndarray
        Zernike expansion coefficients for the phase term.
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    step_f : float
        Step size of at the BFP.
    index_convention : {'standard','fringe'}, optional
        Defines which single index convention to use for the Zernike polynomials.
    defocus_j : int
            Index specifying the defocus term.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, c_A, c_W, aperture_size=1., computation_size=4., 
                 N_pts=128, index_convention='standard'):
        """Constructor.
        
        Parameters
        ----------
        c_A : list or ndarray
            Zernike expansion coefficients for the matrix term.
        c_W : list or ndarray
            Zernike expansion coefficients for the phase term.
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        index_convention : {'standard','fringe'}, optional
            Defines which single index convention to use for the Zernike polynomials.
        """
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)

        self.jmax_list = [len(c_A), len(c_W)+1]
        
        self.c_A = c_A
        self.c_W = c_W
        self.index_convention = index_convention
        
        # self.defocus_j = defocus_j(index_convention)

    def get_pupil_array(self): 

        x, y = self.xy_mesh()
        zernike_seq = zernike_sequence(np.max(self.jmax_list), 
                                        self.index_convention, 
                                        x/self.aperture_size, 
                                        y/self.aperture_size)
        
        Amp = np.sum(zernike_seq[...,:self.jmax_list[0]] 
                * self.c_A, axis=2)        
        #Compute the wavefront          
        W = np.sum(zernike_seq[...,1:self.jmax_list[1]] * self.c_W, axis = 2)
       
        
        return Amp *  np.exp(1j * 2 * np.pi * W)       
    

class ScalarPixels(ScalarWindow):
    """BirefringentWindow subclass for representing pixel-based aberrations.

    UnitaryPixels defines a pupil composed of a general Jones matrix where its elements
    are expanded in terms of the pixels defined by their discretization. 

    Attributes
    ----------
    W : ndarray
        Array representing the scalar aberrations
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
    plot_pupil_field()
        Plots specified components of the array for the pupil.
    """
    def __init__(self, W, qs, aperture_size=1., computation_size=4., 
                 N_pts=128):
        """Constructor.
        
        Parameters
        ----------
        W : ndarray
            Array representing the scalar aberrations
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        ScalarWindow.__init__(self, aperture_size, computation_size, N_pts)
        
        self.W = W
        self.qs = qs

    def get_pupil_array(self): 
        aperture = self.get_aperture(dummy_ind=0)
        return aperture * np.exp(2*np.pi*1j*self.W)
    

# class ApodizedUnitary(BirefringentWindow):
#     '''
#     UnitaryAberrations defines a pupil composed of a general Jones matrix where its elements
#     are expanded in terms of Zernike polynomials. 
#     Note: The piston term controlling the overall amplutde of the mask is fixed to one 
#     (if we want to optimize it we'll need to change it but it is redundant with photobleaching 
#     amplitudes) and the piston and defocus terms of the scalar phase are ommitted.
#     '''
#     def __init__(self, c_A, c_W, c_q, aperture_size=1., computation_size=4., 
#                  N_pts=128, index_convention='standard'):
        
#         BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        
#         # assert len(jmax_list) == 5
#         # self.jmax_list = jmax_list
#         self.jmax_list = []
#         for i in range(4):
#             self.jmax_list += [len(c_q[i])]
#         self.jmax_list += [len(c_W)+2]
#         self.c_q = c_q
#         self.c_W = c_W
#         self.c_A = c_A

#         self.jmax_list += [len(c_A)]
#         self.index_convention = index_convention
        
#         self.defocus_j = defocus_j(index_convention)

#     def get_pupil_array(self): 

#         x, y = self.xy_mesh()
#         N_pts = x.shape[0]
#         zernike_seq = zernike_sequence(np.max(self.jmax_list), 
#                                         self.index_convention, 
#                                         x/self.aperture_size, 
#                                         y/self.aperture_size)
#         # Computes the unitary decomposition matrix and scalar terms         
#         #initialize the matrix term   
#         qs = np.zeros((N_pts,N_pts,4))   
#         for k in range(4):
#             qs[...,k] = np.sum(zernike_seq[...,:self.jmax_list[k]] 
#                 * self.c_q[k], axis=2)        
#         #Compute the matrix term    
#         Q = np.zeros((N_pts,N_pts,2,2), dtype=np.cfloat)
#         Q[...,0,0] = qs[...,0] + 1j * qs[...,3]
#         Q[...,0,1] = qs[...,2] + 1j * qs[...,1]
#         Q[...,1,0] = -np.conj(Q[...,0,1])
#         Q[...,1,1] = np.conj(Q[...,0,0])
#         normQ = np.sum(np.abs(qs)**2, axis=-1)**(1/2)
        
#         (qs[0]**2+qs[1]**2+qs[2]**2+qs[3]**2)**(1/2)
#         normQ[normQ==0] = 1

#         temp = np.hstack((self.c_W[:self.defocus_j-1], [0],
#              self.c_W[self.defocus_j-1:]))     
#         W = np.sum(zernike_seq[...,1:self.jmax_list[-2]] * temp, axis = 2)
#         Amp = np.sum(zernike_seq[...,:self.jmax_list[-1]] * self.c_A, axis = 2)
#         #Compute the scalar term
#         Gamma = np.empty((N_pts,N_pts,1,1), dtype=np.cfloat)
#         Gamma[...,0,0] = Amp * np.exp(1j * 2 * np.pi * W)/normQ
        
#         return Gamma * Q 