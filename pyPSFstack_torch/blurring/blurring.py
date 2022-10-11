import torch
import torch.nn as nn
from math import factorial

from ..diversities.pupil_diversities import torchNoDiversity, torchDerivativeDiversity
from ..blurring.kernels import torchBKSASphere, torchBK2DSphere

class torchBlurring(nn.Module):
    def __init__(self):
        super(torchBlurring, self).__init__()
        
        self.diversity = torchNoDiversity()  

    def compute_blurred_psfs(self, input):
        raise NotImplementedError("Please Implement this method")

    def _compute_intensity(self, input):
        return torch.sum(torch.abs(input)**2,dim=(-2,-1))

class torchNoBlurring(torchBlurring):

    def forward(self, input):
        return self._compute_intensity(input)

class torch2DBlurring(torchBlurring):

    def __init__(self,
                 radius=0.,
                 emission='sphere',
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 ):

        super(torch2DBlurring, self).__init__()


        if emission == 'sphere':
            self.bk = torchBK2DSphere(
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts)

    def forward(self, input):
        output = self._compute_intensity(input)
        output = self.bk(output)
        return torch.abs(output)


class torchSABlurring(torchBlurring):

    def __init__(self,
                 radius=0.,
                 emission='sphere',
                 ni=1.33,
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 ):

        super(torchSABlurring, self).__init__()

        self.diversity = torchDerivativeDiversity(
                                    ni=ni, 
                                    nf=nf, 
                                    aperture_size=aperture_size, 
                                    computation_size=computation_size, 
                                    N_pts=N_pts)

        if emission == 'sphere':
            self.bk = torchBKSASphere(
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts)

    def forward(self, input):
        output = self._compute_0N2_intensity_der(input)
        output = self.bk(output)
        return torch.abs(output)

    def _compute_0N2_intensity_der(self, input):
    
        field_m = input
        in_sh = list(input.shape)
        l_max = 2
        int_m = torch.zeros(in_sh[:2]+[2]+in_sh[3:], dtype=torch.cfloat)

        int_m[:,:,0,...] = torch.abs(field_m[:,:,0,...])**2
        int_m[:,:,1,...] = torch.conj(field_m[:,:,0,...]) * field_m[:,:,2,...] \
                            + 2 *  torch.abs(field_m[:,:,1,...])**2  \
                            + field_m[:,:,0,...] * torch.conj(field_m[:,:,2,...] )
        
        return torch.sum(int_m, dim=(-2,-1))
        