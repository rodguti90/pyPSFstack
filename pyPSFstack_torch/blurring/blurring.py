import torch
import torch.nn as nn
from math import factorial

from ..diversities.pupil_diversities import torchNoDiversity, torchDerivativeDiversity
from ..blurring.kernels import torchBKSASphere

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

class torchSABlurring(torchBlurring):

    def __init__(self,
                 radius=0.,
                 m=2,
                 emission='sphere',
                 ni=1.33,
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 ):

        super(torchSABlurring, self).__init__()

        self.diversity = torchDerivativeDiversity(m=m, 
                                    ni=ni, 
                                    nf=nf, 
                                    aperture_size=aperture_size, 
                                    computation_size=computation_size, 
                                    N_pts=N_pts)
        self.m_max = m

        if emission == 'sphere':
            self.bk = torchBKSASphere(m,
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts)

    def forward(self, input):
        output = self._compute_even_intensity_der(input)
        output = self.bk(output)
        return torch.abs(output)

    def _compute_even_intensity_der(self, input, orientation):
    
    
        field_m = input
        
        in_sh = list(input.shape)
        l_max = self.m_max//2 +1
        int_m = torch.zeros(in_sh[:2]+[l_max]+in_sh[3:], dtype=torch.cfloat)

        for l in range(l_max):
            for k in range(2*l+1):
                int_m[:,:,l,...] += (factorial(2*l)/(factorial(k)*factorial(2*l-k))) * \
                    torch.conj(field_m[:,:,k,...]) * field_m[:,:,2*l-k,...]
        
        return torch.sum(int_m, dim=(-2,-1))
        