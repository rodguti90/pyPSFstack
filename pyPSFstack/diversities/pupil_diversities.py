import numpy as np
from ..pupil import PupilDiversity

class NoDiversity():
    def forward(self, input):
        return input

class ZDiversity(PupilDiversity):

    def __init__(self, 
                 z_list, 
                 nf=1.518, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
                 
        PupilDiversity.__init__(self, aperture_size, computation_size, N_pts)
        
        self.z_list = np.reshape(np.array(z_list), (1,1,-1))
        self.N_zdiv = len(z_list)
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).astype(np.cfloat)
        return np.exp(1j*2*np.pi*self.nf*self.z_list*(1-ur**2)**(1/2))
    

class DDiversity(PupilDiversity):

    def __init__(self, 
                 diff_del_list, 
                 ni=1.33, 
                 nf=1.518, 
                 aperture_size = 1., 
                 computation_size=4., 
                 N_pts=128):

        PupilDiversity.__init__(self, aperture_size, computation_size, N_pts)
        
        self.diff_del_list = np.reshape(np.array(diff_del_list), (1,1,-1))
        self.N_ddiv = len(diff_del_list)
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).astype(np.cfloat)
        return np.exp(1j*2*np.pi*self.diff_del_list*self.ni
                      *(1-(self.nf*ur/self.ni)**2)**(1/2)) 

class DerivativeDiversity(PupilDiversity):

    def __init__(self, 
                 m=2, 
                 ni=1.33, 
                 nf=1.518, 
                 aperture_size = 1., 
                 computation_size=4., 
                 N_pts=128):

        PupilDiversity.__init__(self, aperture_size, computation_size, N_pts)
        
        self.m = m
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        nr, nc = ur.shape
        ur = ur.astype(np.cfloat)
        pupil_array = np.empty((nr,nc,self.m+1), dtype=np.cfloat)
        for l in range(self.m+1):
            pupil_array[...,l] = (1j*2*np.pi*self.ni*(1-(self.nf*ur/self.ni)**2)**(1/2))**l
        
        return pupil_array