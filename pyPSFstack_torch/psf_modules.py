import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from .pupils.sources import torchDipoleInterfaceSource
from .pupils.windows import SEO
from .pupils.aberrations import torchUnitaryAberrations
# from .diversities.diversities import PDiversity#, ZDiversity
from .diversities.pupil_diversities import torchZDiversity
from .functions import crop_center

class torchPSFStack(nn.Module):

    def __init__(self,
                 pupils,
                 zdiversity=None,
                 pdiversity=None,
                 ):
        super(torchPSFStack, self).__init__()
        
        self.pupils = pupils
        self.zdiversity = zdiversity
        self.pdiversity = pdiversity
        self.N_pupils = len(self.pupils)
        self.N_pts = self.pupils[0].N_pts

    def forward(self, N_data):

        output = self.pupils[0]()
        for ind in range(self.N_pupils-1):
            output = self.pupils[ind+1](output)

        output = self.zdiversity(output)

        output = fft.fftshift(
            fft.fft2(output,
                     dim=(0,1),
                     s=(self.N_pts,self.N_pts)),
            dim=(0,1))/self.N_pts

        output = crop_center(output, N_data)

        if self.pdiversity is not None:
            output = self.pdiversity.forward(output)
        output = torch.sum(torch.abs(output)**2,dim=(-2,-1))

        return output

class PhotoBleachBackground(nn.Module):
    def __init__(self, sh, b_est=0):
        b = b_est * torch.ones(sh, dtype=torch.float)
        self.b = nn.Parameter(b, requires_grad=True)
        self.a = nn.Parameter(torch.ones(sh, requires_grad=True, dtype=torch.float))

    def forward(self, input):
        return self.a * input + self.b

# class torchPSFStack(nn.Module):

#     def __init__(self,
#                  pdiversity=None,
#                  aperture_size=.99, 
#                  computation_size=4., 
#                  N_pts=128, 
#                  ni=1.33, 
#                  nf=1.518, 
#                  delta=0.1,
#                  jmax=[15]*5, 
#                  index_convention='fringe', z_list=None
#                  ):
#         super(torchPSFStack, self).__init__()
        
#         self.N_pts = N_pts

#         self.source = torchDipoleInterfaceSource(
#             aperture_size=aperture_size,
#             computation_size=computation_size, 
#             N_pts=N_pts, 
#             ni=ni, 
#             nf=nf, 
#             delta=delta)

#         self.aberrations = torchUnitaryAberrations(
#             aperture_size=aperture_size, 
#             computation_size=computation_size, 
#             N_pts=N_pts, 
#             jmax=jmax, 
#             index_convention=index_convention)

#         self.zdiversity = torchZDiversity(z_list,
#             aperture_size=aperture_size, 
#             computation_size=computation_size, 
#             N_pts=N_pts, 
#             nf=nf)

#         self.pdiversity = pdiversity

#     def forward(self, N_data):

#         input = self.source()
#         input = self.aberrations(input)
#         input = self.zdiversity(input)
#         input = fft.fftshift(
#             fft.fft2(input,
#                      dim=(0,1),
#                      s=(self.N_pts,self.N_pts)),
#             dim=(0,1))

#         input = crop_center(input, N_data)

#         if self.pdiversity is not None:
#             input = self.pdiversity.forward(input)
#         output = torch.sum(torch.abs(input)**2,dim=(-2,-1))

#         return output



class IncoherentPSF(nn.Module):

    def __init__(self,
                 pdiversity=None,
                 aperture_size=.99, 
                 computation_size=4., 
                 N_pts=128, 
                 ni=1.33, 
                 nf=1.518, 
                 delta=0.1,
                 c=1.24*np.pi,
                 jmax=[15]*5, 
                 index_convention='fringe',
                 z_list=None
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

        self.zdiversity = torchZDiversity(z_list,
            aperture_size=aperture_size, 
            computation_size=computation_size, 
            N_pts=N_pts, 
            nf=nf)

        self.pdiversity = PDiversity(pol_analyzer, angle_list=angle_list)

    def forward(self, N_data):

        input = self.source()
        input = self.window(input)
        input = self.aberrations(input)
        input = self.zdiversity(input)
        input = fft.fftshift(
            fft.fft2(input,
                     dim=(0,1),
                     s=(self.N_pts,self.N_pts)),
            dim=(0,1))
        # input = fft.fftshift(
        #     fft.fft2(fft.ifftshift(input,dim=(0,1)),
        #              dim=(0,1),
        #              s=(self.N_pts,self.N_pts)),
        #     dim=(0,1))
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
