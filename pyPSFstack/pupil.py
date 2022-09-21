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
    
    def xy_mesh(self):
        u_vec = np.arange(-1, 1 + self.step_f, self.step_f)
        return np.meshgrid(u_vec,u_vec)
    
    def polar_mesh(self):
        ux, uy = self.xy_mesh()
        ur = np.sqrt(ux**2 + uy**2)
        uphi = np.arctan2(uy, ux)
        return ur, uphi

    def get_aperture(self):
        ur, _ = self.polar_mesh()
        n_pts = ur.shape[0]
        aperture = np.empty((n_pts,n_pts,1,1), dtype=np.cfloat)
        aperture[...,0,0] = ur**2 <= self.aperture_size**2
        return aperture

    def get_pupil_array(self):       
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




              