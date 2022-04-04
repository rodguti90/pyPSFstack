import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from .sources import DipoleInterfaceSource
from .windows import SEO
from .aberrations import UnitaryPolarizationAberrations
from .diversities import ZDiversity, PDiversity
from .functions import crop_center

class IncoherentPSF(nn.Module):

    def __init__(self,
                 pol_analyzer,
                 angle_list,
                 aperture_size=.99, 
                 computation_size=4., 
                 N_pts=128, 
                 ni=1.33, 
                 nf=1.518, 
                 delta=0.1,
                 c=1.24*np.pi,
                 jmax=[15]*5, 
                 index_convention='fringe'
                 ):
        super(IncoherentPSF, self).__init__()
        
        self.N_pts = N_pts

        self.source = DipoleInterfaceSource(
            aperture_size=aperture_size,
            computation_size=computation_size, 
            N_pts=N_pts, 
            ni=ni, 
            nf=nf, 
            delta=delta)

        self.window = SEO(aperture_size=aperture_size, 
                          computation_size=computation_size, 
                          N_pts=N_pts, 
                          c=c)

        self.aberrations = UnitaryPolarizationAberrations(
            aperture_size=aperture_size, 
            computation_size=computation_size, 
            N_pts=N_pts, 
            jmax=jmax, 
            index_convention=index_convention)

        self.zdiversity = ZDiversity(
            aperture_size=aperture_size, 
            computation_size=computation_size, 
            N_pts=N_pts, 
            nf=nf)

        self.pdiversity = PDiversity(pol_analyzer, angle_list=angle_list)

    def forward(self, z_list, N_data):

        input = self.source()
        input = self.window(input)
        input = self.aberrations(input)
        input = self.zdiversity(input, z_list)
        input = fft.fftshift(
            fft.fft2(fft.ifftshift(input,dim=(0,1)),
                     dim=(0,1),
                     s=(self.N_pts,self.N_pts)),
            dim=(0,1))
        input = crop_center(input, N_data)
        input = self.pdiversity(input)
        output = torch.sum(torch.abs(input)**2,dim=(-2,-1))

        return output





# class ScalarPSF(nn.Module):
#     def __init__(self,aperture_size=.99,computation_size=4., 
#                  N_pts=128, N_out=30, jmax=15, index_convention='fringe',
#                  initial_coefs=None):
#         super(ScalarPSF, self).__init__()

#         self.aberrations = ScalarPhaseAberrations(aperture_size=aperture_size,
#                                                   computation_size=computation_size, 
#                                                   N_pts=N_pts,
#                                                   jmax=jmax,
#                                                   index_convention=index_convention,
#                                                   initial_coefs=initial_coefs)

#         self.ur, _ = polar_mesh(computation_size, 
#                  N_pts)
#         self.aperture = self.ur**2 <= aperture_size**2
#         self.ur = self.ur * self.aperture
#         self.ur = self.ur[...,None]
#         self.N_out = N_out

#     def forward(self, input_zdiv):
        
#         zdiv = torch.reshape(input_zdiv,(1,1,-1))
#         phase_div =torch.exp(1j*2*np.pi*zdiv*(1-self.ur**2)**(1/2))
#         aberrated_pupil = self.aberrations(self.aperture)
#         complex_psfs = propagate_image_plane(aberrated_pupil[...,None]*phase_div)
#         crop_psfs = crop_center(complex_psfs, self.N_out)
#         return torch.abs(crop_psfs)

# class ScalarPhaseAberrations(nn.Module):

#     def __init__(self, aperture_size=.99, computation_size=4., 
#                  N_pts=128, jmax=15, index_convention='fringe',
#                  initial_coefs=None):
#         super(ScalarPhaseAberrations, self).__init__()
        
#         x, y = xy_mesh(computation_size, N_pts)
#         if initial_coefs == 'random':
#             init_coefs = 0.1*torch.randn(jmax-2, requires_grad=True, dtype=torch.float) 
#         else:
#             init_coefs = torch.zeros(jmax-2, requires_grad=True, dtype=torch.float)
        
#         self.c_W = nn.Parameter(init_coefs)
#         self.zernike_seq = zernike_sequence(jmax, 
#                                             index_convention, 
#                                             x/aperture_size, 
#                                             y/aperture_size)
#         self.aperture = x**2 + y**2 <= aperture_size**2
#         self.defocus_j = defocus_j(index_convention)
    
#     def forward(self, input):
        
#         W = torch.sum(self.zernike_seq[...,1:self.defocus_j]*self.c_W[:self.defocus_j-1],-1)\
#             + torch.sum(self.zernike_seq[...,self.defocus_j+1:]*self.c_W[self.defocus_j-1:],-1)
#         Gamma = self.aperture*torch.exp(1j*2*np.pi*W)
#         return input * Gamma
