import torch
import torch.nn as nn


class torchPupil(nn.Module):
    def __init__(self, aperture_size, computation_size, N_pts):
        super(torchPupil, self).__init__()
        self.aperture_size = aperture_size 
        self.computation_size = computation_size
        self.N_pts = N_pts
        self.step_f = self.computation_size/self.N_pts

    def xy_mesh(self, umax=1):
        u_vec = torch.arange(-umax,
                        umax,
                        self.step_f, dtype = torch.float32)
        uy, ux = torch.meshgrid(u_vec,u_vec)
        return ux, uy

    def polar_mesh(self, umax=1):
        ux, uy = self.xy_mesh(umax=umax)
        ur = torch.sqrt(ux**2 + uy**2)
        uphi = torch.atan2(uy, ux)
        return ur, uphi
    
    def get_aperture(self, umax=1, dummy_ind=2):
        ur, _ = self.polar_mesh(umax=umax)
        n_pts = ur.shape[0]
        aperture = torch.empty([n_pts,n_pts]+[1]*dummy_ind, dtype=torch.float)
        aperture = torch.reshape(ur**2, [n_pts,n_pts]+[1]*dummy_ind) \
            <= self.aperture_size**2
        return aperture

    def get_pupil_array(self):       
        raise NotImplementedError("Please Implement this method")

class torchPupilDiversity(torchPupil):
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchPupilDiversity, self).__init__(aperture_size, computation_size, N_pts)

    def forward(self, input):
        pupil_array = self.get_pupil_array()
        dims = len(input.shape)
        sh = list(pupil_array.shape)
        output = input[..., None, :, :] \
            * torch.reshape(pupil_array, sh[:2]+[1]*(dims-4)+[sh[2]]+[1,1])
        return output

class torchSource(torchPupil):
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchSource, self).__init__(aperture_size, computation_size, N_pts)

    def forward(self):
        return self.get_pupil_array()

class torchBirefringentWindow(torchPupil):
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchBirefringentWindow, self).__init__(aperture_size, computation_size, N_pts)

    def forward(self, input):
        return self.get_pupil_array() @ input

class torchBlurringKernel(torchPupil):
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchBlurringKernel, self).__init__(aperture_size, computation_size, N_pts)

    
    def forward(self, input):
        bk = self.get_pupil_array()
        dim_bk = len(bk.shape)
        dim_in = len(input.shape)
        assert input.shape[:dim_bk] == bk.shape

        bk = torch.reshape(bk, list(len(bk.shape))+[1]*(dim_in-dim_bk))
        otf = torch.fft.fftshift(torch.fft.ifft2(input, axes=(0,1)), axes=(0,1))
        output = torch.fft.fft2(torch.fft.ifftshift(otf * bk, axes=(0,1)), 
                axes=(0,1))
        
        if dim_bk==3:
            output = torch.sum(output, axis=2)

        return output