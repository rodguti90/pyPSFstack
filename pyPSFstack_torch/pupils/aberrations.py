import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from ..pupil import torchBirefringentWindow
from .zernike_functions import zernike_sequence, defocus_j

class torchUnitaryAberrations(torchBirefringentWindow):

    def __init__(self, aperture_size=.99, computation_size=4., 
                 N_pts=128, jmax_list=[15]*5, index_convention='fringe'
                 ):
        super(torchUnitaryAberrations, self).__init__(aperture_size, computation_size, N_pts)
        
        tempq = torch.zeros((jmax_list[0]), dtype=torch.float)
        tempq[0] = 1
        c_q = [nn.Parameter(tempq, requires_grad=True)]
        for i in range(1,4):
            c_q += [nn.Parameter(torch.zeros(jmax_list[i], requires_grad=True, dtype=torch.float))]
        self.c_q = nn.ParameterList(c_q)
        
        self.c_W = nn.Parameter(torch.zeros(jmax_list[4]-2, requires_grad=True, dtype=torch.float))
        self.jmax=jmax_list
        self.index_convention = index_convention                  
        self.defocus_j = defocus_j(index_convention)
    
    def get_pupil_array(self):
        ux, uy = self.xy_mesh()
        N_pupil = ux.shape[0]
        zernike_seq = zernike_sequence(np.max(self.jmax), 
                                        self.index_convention, 
                                        ux/self.aperture_size, 
                                        uy/self.aperture_size)

        qs = torch.zeros((4,N_pupil,N_pupil),  dtype=torch.cfloat)
        for q_ind in range(4):
            qs[q_ind] = torch.sum(self.c_q[q_ind]*zernike_seq[...,:self.jmax[q_ind]],-1)
        Q = torch.zeros((N_pupil,N_pupil,2,2), dtype=torch.cfloat)
        Q[...,0,0] = qs[0] + 1j*qs[3]
        Q[...,0,1] = qs[2] + 1j*qs[1]
        Q[...,1,0] = -qs[2] + 1j*qs[1]
        Q[...,1,1] = qs[0] - 1j*qs[3]

        W = torch.sum(zernike_seq[...,1:self.defocus_j]*self.c_W[:self.defocus_j-1],-1)\
            + torch.sum(zernike_seq[...,self.defocus_j+1:self.jmax[4]]*
                        self.c_W[self.defocus_j-1:],-1)
        Gamma = self.get_aperture(dummy_ind=0)*torch.exp(1j*2*np.pi*W)

        return ( Gamma[...,None,None] * Q)