import matplotlib.pyplot as plt
import numpy as np
from .functions import colorize

class Pupil():
    
    def __init__(self, aperture_size, computation_size, N_pts):

        self.aperture_size = aperture_size 
        self.computation_size = computation_size
        self.N_pts = N_pts
        self.step_f = self.computation_size/self.N_pts
        # self.set_aperture()

    # def step_fourier(self):
    #     return self.computation_size/self.N_pts
    
    def xy_mesh(self, umax=1):
        u_vec = np.arange(-umax, umax, self.step_f)
        return np.meshgrid(u_vec,u_vec)
    
    def polar_mesh(self, umax=1):
        ux, uy = self.xy_mesh(umax=umax)
        ur = np.sqrt(ux**2 + uy**2)
        uphi = np.arctan2(uy, ux)
        return ur, uphi

    def get_aperture(self, umax=1, dummy_ind=2):
        ur, _ = self.polar_mesh(umax=umax)
        n_pts = ur.shape[0]
        aperture = np.empty([n_pts,n_pts]+[1]*dummy_ind, dtype=np.cfloat)
        aperture = np.expand_dims(ur, list(np.arange(2,dummy_ind+2)))**2 \
            <= self.aperture_size**2
        return aperture

    def get_pupil_array(self):       
        raise NotImplementedError("Please Implement this method")
    
    def forward(self):       
        raise NotImplementedError("Please Implement this method")

    def plot_pupil_field(self):
        pupil = self.get_pupil_array()
        sh = pupil.shape
        if len(sh) == 4:
            fig, axs = plt.subplots(sh[-2], sh[-1], figsize=(4*sh[-1],4*sh[-2]))
            for r in range(sh[-2]):
                for c in range(sh[-1]):
                    axs[r,c].imshow(colorize(pupil[...,r,c]))
                    axs[r,c].set_axis_off()


class PupilDiversity(Pupil):
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
                 
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
    
    def forward(self, input):
        pupil_array = self.get_pupil_array()
        dims = len(input.shape)
        output = input[..., None, :, :] \
            * np.expand_dims(pupil_array,list(np.arange(2,dims-2,1))+[-2,-1])
        return output

class BirefringentWindow(Pupil):
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
                 
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
    
    def forward(self, input):
        return self.get_pupil_array() @ input
              
class BlurringKernel(Pupil):
    def __init__(self,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):

        Pupil.__init__(self, aperture_size, computation_size, N_pts)
    
    def forward(self, input):
        bk = self.get_pupil_array()
        dim_bk = len(bk.shape)
        dim_in = len(input.shape)
        # assert input.shape[:dim_bk] == bk.shape

        bk = np.expand_dims(bk, list(np.arange(dim_bk,dim_in)))
        otf = np.fft.fftshift(np.fft.ifft2(input, axes=(0,1)), axes=(0,1))
        output = np.fft.fft2(np.fft.ifftshift(otf * bk, axes=(0,1)), 
                axes=(0,1))
        
        if dim_bk==3:
            output = np.sum(output, axis=2)

        return output