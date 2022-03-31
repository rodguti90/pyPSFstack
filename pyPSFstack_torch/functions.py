import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
from .zernike_functions import zernike_sequence, defocus_j

def cart2pol(x,y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return rho, phi

def xy_mesh(size, step):
    u_vec = torch.arange(-size/2,
                    size/2,
                    step, dtype = torch.float32)
    return torch.meshgrid(u_vec,u_vec)

def polar_mesh(size, step):
    ux, uy = xy_mesh(size, step)
    ur = torch.sqrt(ux**2 + uy**2)
    uphi = torch.atan2(uy, ux)
    return ur, uphi

def crop_center(input, size):
    x = input.shape[0]
    y = input.shape[1]
    start_x = x//2-(size//2)
    start_y = y//2-(size//2)
    return input[start_x:start_x+size,start_y:start_y+size,...]

def set_aperture(self):
    ur, _ = self.polar_mesh()
    self.aperture = np.empty((self.N_pts,self.N_pts,1,1), dtype=np.cfloat)
    self.aperture[...,0,0] = ur**2 <= self.aperture_size**2

def propagate_image_plane(input, N_comp):
    output = fft.fftshift(
            fft.fft2(
                fft.ifftshift(input, s=N_comp, dim=(0,1)), 
                dim=(0,1)), 
            dim=(0,1))
    return output







class ScalarPSF(nn.Module):
    def __init__(self,aperture_size=.99,computation_size=4., 
                 N_pts=128, N_out=30, jmax=15, index_convention='fringe',
                 initial_coefs=None):
        super(ScalarPSF, self).__init__()

        self.aberrations = ScalarPhaseAberrations(aperture_size=aperture_size,
                                                  computation_size=computation_size, 
                                                  N_pts=N_pts,
                                                  jmax=jmax,
                                                  index_convention=index_convention,
                                                  initial_coefs=initial_coefs)

        self.ur, _ = polar_mesh(computation_size, 
                 N_pts)
        self.aperture = self.ur**2 <= aperture_size**2
        self.ur = self.ur * self.aperture
        self.ur = self.ur[...,None]
        self.N_out = N_out

    def forward(self, input_zdiv):
        
        zdiv = torch.reshape(input_zdiv,(1,1,-1))
        phase_div =torch.exp(1j*2*np.pi*zdiv*(1-self.ur**2)**(1/2))
        aberrated_pupil = self.aberrations(self.aperture)
        complex_psfs = propagate_image_plane(aberrated_pupil[...,None]*phase_div)
        crop_psfs = crop_center(complex_psfs, self.N_out)
        return torch.abs(crop_psfs)

class ScalarPhaseAberrations(nn.Module):

    def __init__(self, aperture_size=.99, computation_size=4., 
                 N_pts=128, jmax=15, index_convention='fringe',
                 initial_coefs=None):
        super(ScalarPhaseAberrations, self).__init__()
        
        x, y = xy_mesh(computation_size, N_pts)
        if initial_coefs == 'random':
            init_coefs = 0.1*torch.randn(jmax-2, requires_grad=True, dtype=torch.float) 
        else:
            init_coefs = torch.zeros(jmax-2, requires_grad=True, dtype=torch.float)
        
        self.c_W = nn.Parameter(init_coefs)
        self.zernike_seq = zernike_sequence(jmax, 
                                            index_convention, 
                                            x/aperture_size, 
                                            y/aperture_size)
        self.aperture = x**2 + y**2 <= aperture_size**2
        self.defocus_j = defocus_j(index_convention)
    
    def forward(self, input):
        
        W = torch.sum(self.zernike_seq[...,1:self.defocus_j]*self.c_W[:self.defocus_j-1],-1)\
            + torch.sum(self.zernike_seq[...,self.defocus_j+1:]*self.c_W[self.defocus_j-1:],-1)
        Gamma = self.aperture*torch.exp(1j*2*np.pi*W)
        return input * Gamma
