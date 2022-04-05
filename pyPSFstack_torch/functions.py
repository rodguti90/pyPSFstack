import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
# from .zernike_functions import zernike_sequence, defocus_j

def cart2pol(x,y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return rho, phi

def xy_mesh(size, step):
    u_vec = torch.arange(-size/2,
                    size/2,
                    step, dtype = torch.float32)
    uy, ux = torch.meshgrid(u_vec,u_vec)
    return ux, uy

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




