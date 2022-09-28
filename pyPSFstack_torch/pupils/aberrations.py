import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from ..pupil import torchBirefringentWindow
from ..functions import xy_mesh
from .zernike_functions import zernike_sequence, defocus_j

class torchUnitaryAberrations(torchBirefringentWindow):

    def __init__(self, aperture_size=.99, computation_size=4., 
                 N_pts=128, jmax=[15]*2, index_convention='fringe'
                 ):
        super(torchUnitaryAberrations, self).__init__(aperture_size, computation_size, N_pts)

        tempq = torch.zeros((4,jmax[0]), dtype=torch.float)
        tempq[0,0] = 1
        self.c_q = nn.Parameter(tempq, requires_grad=True)
        
        
        self.c_W = nn.Parameter(torch.zeros(jmax[1]-2, requires_grad=True, dtype=torch.float))
        self.jmax=jmax
        step = computation_size/N_pts
        # Limit the pupil to the maximum region of one to avoid wasating memory
        ux, uy = xy_mesh(2, step)
        self.N_pupil = ux.shape[0]
        self.zernike_seq = zernike_sequence(np.max(jmax), 
                                            index_convention, 
                                            ux/aperture_size, 
                                            uy/aperture_size)
        self.aperture = ux**2 + uy**2 <= aperture_size**2
        self.defocus_j = defocus_j(index_convention)
    
    def forward(self, input):
        qs = torch.zeros((4,self.N_pupil,self.N_pupil),  dtype=torch.cfloat)
        for q_ind in range(4):
            qs[q_ind] = torch.sum(self.c_q[q_ind]*self.zernike_seq[...,:self.jmax[0]],-1)
        Q = torch.zeros((self.N_pupil,self.N_pupil,2,2), dtype=torch.cfloat)
        Q[...,0,0] = qs[0] + 1j*qs[3]
        Q[...,0,1] = qs[2] + 1j*qs[1]
        Q[...,1,0] = -qs[2] + 1j*qs[1]
        Q[...,1,1] = qs[0] - 1j*qs[3]

        W = torch.sum(self.zernike_seq[...,1:self.defocus_j]*self.c_W[:self.defocus_j-1],-1)\
            + torch.sum(self.zernike_seq[...,self.defocus_j+1:self.jmax[1]]*self.c_W[self.defocus_j-1:],-1)
        Gamma = self.aperture*torch.exp(1j*2*np.pi*W)

        return ( Gamma[...,None,None] * Q) @ input