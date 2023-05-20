"""Module containing the definitions of all polarization diversities."""
import numpy as np


class PDiversity():
    """PDiversity superclass for defining polarization diversities.
    
    Attributes
    ----------
    jones_list : ndarray
        Array containg the Jones matrices used for the polarization diversity.
    N_pdiv : int
        Number of polarization diversities.
    
    Methods
    -------
    get_jones_list()
        Computes the jones matrices for the polarization diversities.
    """
    def __init__(self, *args):
        self.jones_list = self.get_jones_list(*args)
        self.N_pdiv = len(self.jones_list)

    def get_jones_list(self):
        raise NotImplementedError("Please Implement this method")

    def _forward(self, input):
        return self.jones_list @ input[...,None,:,:]

class PDiversity_QWP(PDiversity):
    """PDiversity subclass for defining polarization diversities based on QWP.
    
    Attributes
    ----------
    jones_list : ndarray
        Array containg the Jones matrices used for the polarization diversity.
    N_pdiv : int
        Number of polarization diversities.
    
    Methods
    -------
    get_jones_list()
        Computes the jones matrices for the polarization diversities.
    """
    def __init__(self, angles):
        """Constructor.
        
        Parameters
        ----------
        angles : list or ndarray
            List with the angles in radians used for the diversity. 
        """
        PDiversity.__init__(self, angles)

    def get_jones_list(self, angles):
        return jones_qwp(angles)

class PDiversity_HWP(PDiversity):
    """PDiversity subclass for defining polarization diversities based on HWP.
    
    Attributes
    ----------
    jones_list : ndarray
        Array containg the Jones matrices used for the polarization diversity.
    N_pdiv : int
        Number of polarization diversities.
    
    Methods
    -------
    get_jones_list()
        Computes the jones matrices for the polarization diversities.
    """
    def __init__(self, angles):
        """Constructor.
        
        Parameters
        ----------
        angles : list or ndarray
            List with the angles in radians used for the diversity. 
        """
        PDiversity.__init__(self, angles)

    def get_jones_list(self, angles):
        return jones_hwp(angles)

class PDiversity_LP(PDiversity):
    """PDiversity subclass for defining polarization diversities based on linear polarizer.
    
    Attributes
    ----------
    jones_list : ndarray
        Array containg the Jones matrices used for the polarization diversity.
    N_pdiv : int
        Number of polarization diversities.
    
    Methods
    -------
    get_jones_list()
        Computes the jones matrices for the polarization diversities.
    """
    def __init__(self, angles):
        """Constructor.
        
        Parameters
        ----------
        angles : list or ndarray
            List with the angles in radians used for the diversity. 
        """
        PDiversity.__init__(self, angles)
        
    def get_jones_list(self, angles):
        return jones_lp(angles)

class PDiversity_GWP(PDiversity):
    """PDiversity subclass for defining polarization diversities based on generalized wave plate.
    
    Attributes
    ----------
    jones_list : ndarray
        Array containg the Jones matrices used for the polarization diversity.
    N_pdiv : int
        Number of polarization diversities.
    
    Methods
    -------
    get_jones_list()
        Computes the jones matrices for the polarization diversities.
    """
    def __init__(self, angles, eta):
        """Constructor.
        
        Parameters
        ----------
        angles : list or ndarray
            List with the angles in radians used for the diversity. 
        eta : float
            Retardance of the generalized wave plate.
        """
        PDiversity.__init__(self, angles, eta)

    def get_jones_list(self, angles, eta):
        return jones_gwp(angles, eta)

class PDiversity_Compound(PDiversity):
    """PDiversity subclass for defining compounded polarization diversities.
    
    Attributes
    ----------
    jones_list : ndarray
        Array containg the Jones matrices used for the polarization diversity.
    N_pdiv : int
        Number of polarization diversities.
    
    Methods
    -------
    get_jones_list()
        Computes the jones matrices for the polarization diversities.
    """
    def __init__(self, pdivs):
        """Constructor.
        
        Parameters
        ----------
        pdivs : list of PDiversity objects
            List of the elementary diversities used to build a compound diversity.
        """
        PDiversity.__init__(self, pdivs)

    def get_jones_list(self, pdivs):
        n_pdivs = len(pdivs)
        jones = pdivs[0].jones_list
        for n_p in range(1,n_pdivs):
            jones = np.expand_dims(pdivs[n_p].jones_list,
                list(np.arange(1,n_p+1))) @ jones
        return jones.reshape(-1,2,2)

def jones_qwp(theta):
    n_a = len(theta)
    jones =np.empty((n_a,2,2), dtype=complex)
    jones[...,0,0] = np.cos(theta)**2 + 1j * np.sin(theta)**2
    jones[...,0,1] = (1-1j)*np.sin(theta)*np.cos(theta)
    jones[...,1,1] = 1j*np.conj(jones[...,0,0])
    jones[...,1,0] = jones[...,0,1]
    return np.exp(-1j*np.pi/4) * jones

def jones_lp(theta):
    n_a = len(theta)
    jones =np.empty((n_a,2,2), dtype=complex)
    jones[...,0,0] = np.cos(theta)**2
    jones[...,0,1] = np.sin(theta)*np.cos(theta)
    jones[...,1,1] = np.sin(theta)**2
    jones[...,1,0] = jones[...,0,1]
    return jones

def jones_hwp(theta):
    n_a = len(theta)
    jones =np.empty((n_a,2,2), dtype=complex)
    jones[...,0,0] = np.cos(theta)**2 - np.sin(theta)**2
    jones[...,0,1] = 2*np.sin(theta)*np.cos(theta)
    jones[...,1,1] = -jones[...,0,0]
    jones[...,1,0] = jones[...,0,1]
    return np.exp(-1j*np.pi/2) * jones

def jones_gwp(theta, eta):
    n_a = len(theta)
    jones =np.empty((n_a,2,2), dtype=complex)
    jones[...,0,0] = np.exp(-1j*eta/2) * np.cos(theta)**2 + \
        np.exp(1j*eta/2) * np.sin(theta)**2
    jones[...,0,1] = np.exp(-1j*eta/2)*(1-np.exp(1j*eta))*np.sin(theta)*np.cos(theta)
    jones[...,1,1] = np.conj(jones[...,0,0])
    jones[...,1,0] = jones[...,0,1]
    return jones

# def quarter2pol(angle_list):
#     n_ang = len(angle_list)
#     quart2pol_analyzer = np.zeros((2*n_ang,2,2), dtype=np.cfloat)
#     quart2pol_analyzer[:n_ang,0,0] = (np.cos(angle_list)**2 + 1j*np.sin(angle_list)**2)
#     quart2pol_analyzer[:n_ang,0,1] = (1-1j)*np.sin(angle_list)*np.cos(angle_list)
#     quart2pol_analyzer[n_ang:,1,0] = quart2pol_analyzer[:n_ang,0,1]
#     quart2pol_analyzer[n_ang:,1,1] = (1j*np.cos(angle_list)**2 + np.sin(angle_list)**2)   
#     return quart2pol_analyzer