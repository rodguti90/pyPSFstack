import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from .functions import xy_mesh, polar_mesh

class ZDiversity(nn.Module):

    def __init__(self, aperture_size=.99, computation_size=4., 
                 N_pts=128, nf=1.518
                 ):
        super(ZDiversity, self).__init__()

        # self.c = nn.Parameter(torch.tensor(c, requires_grad=True, dtype=torch.float))
        self.nf = nf
        self.step = computation_size/N_pts
        # Limit the pupil to the maximum region of one to avoid wasting memory
        ur, _ = polar_mesh(2, self.step)

        self.N_pupil = ur.shape[0]
        self.aperture = ur**2 <= aperture_size**2

    def forward(self, input, z_list):
        # ur, _ = polar_mesh(2, self.step)
        ur = torch.zeros((self.N_pupil,self.N_pupil,1), dtype=torch.cfloat)
        ur[:,:,0], _ = polar_mesh(2, self.step)
        pupil_array = self.aperture[...,None] \
            *torch.exp(1j*2*np.pi*self.nf*z_list[None,None,:]*(1-ur**2)**(1/2)) 
        
        return pupil_array[...,None,None] * input[...,None,:,:]

class PDiversity(nn.Module):

    def __init__(self, pol_analyzer, angle_list=[]):
        super(PDiversity, self).__init__()

        if pol_analyzer == 'quarter2pol':
            self.jones_list = quarter2pol(angle_list)    
        else:
            raise NotImplementedError("The chosen polarization analyzer is not implemented")
        self.N_pdiv = len(self.jones_list)

    def forward(self, input):
        return self.jones_list @ input[...,None,:,:]

def quarter2pol(angles):
    n_ang = len(angles)
    angle_list = torch.tensor(angles.tolist()) 
    quart2pol_analyzer = torch.zeros((2*n_ang,2,2), dtype=torch.cfloat)
    quart2pol_analyzer[:n_ang,0,0] = (torch.cos(angle_list)**2 + 1j*torch.sin(angle_list)**2)
    quart2pol_analyzer[:n_ang,0,1] = (1-1j)*torch.sin(angle_list)*torch.cos(angle_list)
    quart2pol_analyzer[n_ang:,1,0] = quart2pol_analyzer[:n_ang,0,1]
    quart2pol_analyzer[n_ang:,1,1] = (1j*torch.cos(angle_list)**2 + torch.sin(angle_list)**2)   
    return quart2pol_analyzer