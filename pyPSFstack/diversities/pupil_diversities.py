import numpy as np
from ..pupil import Pupil

class ZDiversity(Pupil):
    def __init__(self, z_list, nf=1.518, aperture_size = 1., computation_size=4., 
                 N_pts=128):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        
        self.z_list = np.reshape(np.array(z_list), (1,1,-1))
        self.N_zdiv = len(z_list)
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).astype(np.cfloat)
        return np.exp(1j*2*np.pi*self.nf*self.z_list*(1-ur**2)**(1/2))


class DDiversity(Pupil):
    def __init__(self, diff_del_list, ni=1.33, nf=1.518, aperture_size = 1., computation_size=4., 
                 N_pts=128):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        
        self.diff_del_list = np.reshape(np.array(diff_del_list), (1,1,-1))
        self.N_ddiv = len(diff_del_list)
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).astype(np.cfloat)
        return np.exp(1j*2*np.pi*self.diff_del_list*self.ni
                      *(1-(self.nf*ur/self.ni)**2)**(1/2)) 

