import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from .functions import xy_mesh, polar_mesh

class SEO(nn.Module):

    def __init__(self, aperture_size=.99, computation_size=4., 
                 N_pts=128, c=1.24*np.pi
                 ):
        super(SEO, self).__init__()

        self.c = nn.Parameter(torch.tensor(c, requires_grad=True, dtype=torch.float))
        
        self.step = computation_size/N_pts
        # Limit the pupil to the maximum region of one to avoid wasting memory
        ur = torch.empty((self.N_pupil,self.N_pupil), dtype=torch.cfloat)
        ur[...] = polar_mesh(2, self.step)
        self.N_pupil = ur.shape[0]
        self.aperture = ur**2 <= aperture_size**2

    def forward(self, input):
        ur, uphi = polar_mesh(2, self.step)
        pupil_array = torch.empty((self.N_pupil,self.N_pupil,2,2), dtype=torch.cfloat)
        pupil_array[...,0,0] = torch.cos(self.c*ur/2) -1j*torch.sin(self.c*ur/2)*torch.cos(uphi)
        pupil_array[...,0,1] = -1j*torch.sin(self.c*ur/2)*torch.sin(uphi)
        pupil_array[...,1,0] = self.pupil_array[...,0,1]
        pupil_array[...,1,1] = torch.conj(self.pupil_array[...,0,0])
        pupil_array = self.aperture * pupil_array 
        
        return pupil_array * input