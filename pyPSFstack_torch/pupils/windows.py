import torch
import torch.nn as nn
import torch.fft as fft
from ..pupil import torchBirefringentWindow
import numpy as np


class torchSEO(torchBirefringentWindow):

    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi
                 ):
        super(SEO, self).__init__(aperture_size, computation_size, N_pts)

        self.c = nn.Parameter(torch.tensor(c, requires_grad=True, dtype=torch.float))
        
    def get_pupil_array(self):
        ur, uphi = self.polar_mesh()
        return self.get_aperture() * jones_seo(ur, uphi, c=self.c)


def jones_seo(ur, uphi, c=1.24*np.pi):
    ny, nx = ur.shape
    pupil_array = torch.empty((ny,nx,2,2), dtype=torch.cfloat)
    pupil_array[...,0,0] = torch.cos(c*ur/2) -1j*torch.sin(c*ur/2)*torch.cos(uphi)
    pupil_array[...,0,1] = -1j*torch.sin(c*ur/2)*torch.sin(uphi)
    pupil_array[...,1,0] = pupil_array[...,0,1]
    pupil_array[...,1,1] = torch.conj(pupil_array[...,0,0])
    
    return pupil_array 