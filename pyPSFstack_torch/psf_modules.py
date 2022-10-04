import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from .functions import crop_center

class torchPSFStack(nn.Module):

    def __init__(self,
                 N_data,
                 pupils,                
                 zdiversity=None,
                 pdiversity=None
                 ):
        super(torchPSFStack, self).__init__()
        
        self.N_data = N_data
        self.pupils = nn.ModuleList(pupils)
        self.zdiversity = zdiversity
        self.pdiversity = pdiversity
        
        self.N_pupils = len(self.pupils)
        self.N_pts = self.pupils[0].N_pts
        
        self.pb_bck = None
        self.scale_factor = 1

    def set_pb_bck(self, bck, opt_b=False, opt_a=False):
        div_shape = []
        if self.zdiversity is not None:
            div_shape += [self.zdiversity.N_zdiv]
        if self.pdiversity is not None:
            div_shape += [self.pdiversity.N_pdiv]
        self.pb_bck = PhotoBleachBackground(div_shape, 
                                            b_est=bck,
                                            opt_a=opt_a,
                                            opt_b=opt_b)

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    def forward(self):

        output = self.scale_factor**(1/2) * self.pupils[0]()
        for ind in range(self.N_pupils-1):
            output = self.pupils[ind+1](output)

        if self.zdiversity is not None:
            output = self.zdiversity(output)

        output = fft.fftshift(
            fft.fft2(output,
                     dim=(0,1),
                     s=(self.N_pts,self.N_pts)),
            dim=(0,1))/self.N_pts

        output = crop_center(output, self.N_data)

        if self.pdiversity is not None:
            output = self.pdiversity.forward(output)
        output = torch.sum(torch.abs(output)**2,dim=(-2,-1))
        
        if self.pb_bck is not None:
            output = self.pb_bck(output)

        return output

class PhotoBleachBackground(nn.Module):
    def __init__(self, sh, b_est=0, opt_b=True, opt_a=True):
        super(PhotoBleachBackground, self).__init__()

        b = b_est * torch.ones(sh, dtype=torch.float)
        if opt_b:
            self.b = nn.Parameter(b, requires_grad=True)
        else: 
            self.b = b_est
        if opt_a:
            self.a = nn.Parameter(torch.ones(sh, requires_grad=True, dtype=torch.float))
        else:
            self.a = 1

    def forward(self, input):
        return self.a * input + self.b




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
