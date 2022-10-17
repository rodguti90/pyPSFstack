from os import urandom
import torch
import torch.nn as nn
import torch.fft as fft
from ..pupil import torchBirefringentWindow, torchScalarWindow
import numpy as np


class torchDefocus(torchScalarWindow):
    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, nf=1.518, delta_z=0.
                 ):
        super(torchDefocus, self).__init__(aperture_size, computation_size, N_pts)
        
        self.nf = nf
        self.delta_z = nn.Parameter(torch.tensor(delta_z, requires_grad=True, dtype=torch.float))
        
    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = ur.type(torch.cfloat)
        aperture = self.get_aperture(dummy_ind=0)
        defocus = torch.exp(1j*2*np.pi*self.nf*self.delta_z*(1-ur**2)**(1/2))
        return aperture * defocus

class torchSAF(torchScalarWindow):
    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, ni=1.33, nf=1.518
                 ):
        super(torchSAF, self).__init__(aperture_size, computation_size, N_pts)
        
        self.ni = ni
        self.nf = nf
        self.delta_d = nn.Parameter(torch.tensor(0, requires_grad=True, dtype=torch.float))
        
    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = ur.type(torch.cfloat)
        aperture = self.get_aperture(dummy_ind=0)
        saf = torch.exp(1j*2*np.pi*self.delta_d*self.ni
                        *(1-(ur*self.nf/self.ni)**2)**(1/2))
        return aperture * saf


class torchSEO(torchBirefringentWindow):

    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi, phi=0, center=[0,0]
                 ):
        super(torchSEO, self).__init__(aperture_size, computation_size, N_pts)

        self.c = nn.Parameter(torch.tensor(c, requires_grad=True, dtype=torch.float))
        self.phi = nn.Parameter(torch.tensor(phi, requires_grad=True, dtype=torch.float))
        self.center= nn.Parameter(torch.tensor(center, requires_grad=True, dtype=torch.float))

    def get_pupil_array(self):
        ux, uy = self.xy_mesh()
        return self.get_aperture() * jones_seo(ux, uy, c=self.c, phi=self.phi, center=self.center)

class torchQplate(torchBirefringentWindow):

    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, q=1, alpha=0
                 ):
        super(torchQplate, self).__init__(aperture_size, computation_size, N_pts)

        self.q = nn.Parameter(torch.tensor(q, requires_grad=True, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True, dtype=torch.float))

    def get_pupil_array(self):
        ur, uphi = self.polar_mesh()
        return self.get_aperture() * jones_seo(ur, uphi, c=self.c)


def jones_qplate(uphi, q, alpha):
    ny, nx = uphi.shape
    pupil_array = torch.empty((ny,nx,2,2), dtype=torch.cfloat)
    theta = q*uphi + alpha
    pupil_array[...,0,0] = 1j*torch.cos(2*theta)
    pupil_array[...,0,1] = 1j*torch.sin(2*theta)
    pupil_array[...,1,0] = -torch.conj(pupil_array[...,0,1])
    pupil_array[...,1,1] = torch.conj(pupil_array[...,0,1])

def jones_seo(ux, uy, c=1.24*np.pi, phi=0, center=torch.tensor([0,0])):
    ny, nx = ux.shape
    uxt = ux - center[0]
    uyt = uy - center[1]
    ur = torch.sqrt(uxt**2 + uyt**2)
    uphi = torch.atan2(uyt, uxt)
    pupil_array = torch.empty((ny,nx,2,2), dtype=torch.cfloat)
    pupil_array[...,0,0] = torch.cos(c*ur/2) -1j*torch.sin(c*ur/2)*torch.cos(uphi-2*phi)
    pupil_array[...,0,1] = -1j*torch.sin(c*ur/2)*torch.sin(uphi-2*phi)
    pupil_array[...,1,0] = pupil_array[...,0,1]
    pupil_array[...,1,1] = torch.conj(pupil_array[...,0,0])
    
    return pupil_array 