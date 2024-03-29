"""Module containing the definitions of all torch polarization diversities."""
import torch


class torchPDiversity():
    """torchPDiversity superclass for defining polarization diversities.
    
    Attributes
    ----------
    jones_list : Tensor
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

    def forward(self, input):
        return self.jones_list @ input[...,None,:,:]

class torchPDiversity_QWP(torchPDiversity):
    """torchPDiversity subclass for defining polarization diversities based on QWP.
    
    Attributes
    ----------
    jones_list : Tensor
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
        angles : list, ndarray, Tensor
            List with the angles in radians used for the diversity. 
        """
        torchPDiversity.__init__(self, angles)

    def get_jones_list(self, angles):
        return jones_qwp(angles)

class torchPDiversity_HWP(torchPDiversity):
    """torchPDiversity subclass for defining polarization diversities based on HWP.
    
    Attributes
    ----------
    jones_list : Tensor
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
        angles : list ndarray, or Tensor
            List with the angles in radians used for the diversity. 
        """
        torchPDiversity.__init__(self, angles)

    def get_jones_list(self, angles):
        return jones_hwp(angles)

class torchPDiversity_GWP(torchPDiversity):
    """torchPDiversity subclass for defining polarization diversities based on generalized wave plate.
    
    Attributes
    ----------
    jones_list : Tensor
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
        angles : list, ndarray, or Tensor
            List with the angles in radians used for the diversity. 
        eta : float
            Retardance of the generalized wave plate.
        """
        torchPDiversity.__init__(self, angles, eta)

    def get_jones_list(self, angles, eta):
        return jones_gwp(angles, eta)

class torchPDiversity_LP(torchPDiversity):
    """torchPDiversity subclass for defining polarization diversities based on linear polarizer.
    
    Attributes
    ----------
    jones_list : Tensor
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
        angles : list, ndarray, or Tensor
            List with the angles in radians used for the diversity. 
        """
        torchPDiversity.__init__(self, angles)
        
    def get_jones_list(self, angles):
        return jones_lp(angles)

class torchPDiversity_Compound(torchPDiversity):
    """torchPDiversity subclass for defining compounded polarization diversities.
    
    Attributes
    ----------
    jones_list : Tensor
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
        pdivs : list of torchPDiversity objects
            List of the elementary diversities used to build a compound diversity.
        """
        torchPDiversity.__init__(self, pdivs)

    def get_jones_list(self, pdivs):
        n_pdivs = len(pdivs)
        jones = pdivs[0].jones_list
        for n_p in range(1,n_pdivs):
            sh = list(pdivs[n_p].jones_list.shape)
            jones = torch.reshape(pdivs[n_p].jones_list, [sh[0]]+[1]*n_p+sh[1:])\
                @ jones
        return jones.reshape(-1,2,2)

def jones_qwp(angles):
    theta = torch.as_tensor(angles)
    n_a = len(theta)
    jones = torch.empty((n_a,2,2), dtype=torch.cfloat)
    jones[...,0,0] = torch.cos(theta)**2 + 1j * torch.sin(theta)**2
    jones[...,0,1] = (1-1j)*torch.sin(theta)*torch.cos(theta)
    jones[...,1,1] = 1j*torch.conj(jones[...,0,0])
    jones[...,1,0] = jones[...,0,1]
    return torch.exp(torch.tensor(-1j*torch.pi/4)) * jones

def jones_lp(angles):
    theta = torch.as_tensor(angles)
    n_a = len(theta)
    jones = torch.empty((n_a,2,2), dtype=torch.cfloat)
    jones[...,0,0] = torch.cos(theta)**2
    jones[...,0,1] = torch.sin(theta)*torch.cos(theta)
    jones[...,1,1] = torch.sin(theta)**2
    jones[...,1,0] = jones[...,0,1]
    return jones

def jones_hwp(angles):
    theta = torch.as_tensor(angles)
    n_a = len(theta)
    jones = torch.empty((n_a,2,2), dtype=torch.cfloat)
    jones[...,0,0] = torch.cos(theta)**2 - torch.sin(theta)**2
    jones[...,0,1] = 2*torch.sin(theta)*torch.cos(theta)
    jones[...,1,1] = -jones[...,0,0]
    jones[...,1,0] = jones[...,0,1]
    return torch.exp(torch.tensor(-1j*torch.pi/2)) * jones

def jones_gwp(angles, eta):
    theta = torch.as_tensor(angles)
    eta = torch.as_tensor(eta)
    n_a = len(theta)
    jones = torch.empty((n_a,2,2), dtype=torch.cfloat)
    jones[...,0,0] = torch.exp(-1j*eta/2) * torch.cos(theta)**2 + \
        torch.exp(1j*eta/2) * torch.sin(theta)**2
    jones[...,0,1] = torch.exp(-1j*eta/2)*(1-torch.exp(1j*eta))*torch.sin(theta)*torch.cos(theta)
    jones[...,1,1] = torch.conj(jones[...,0,0])
    jones[...,1,0] = jones[...,0,1]
    return jones