import torch as torch
import torch.nn as nn
from ..pupil import torchPupilDiversity


class torchNoDiversity(nn.Module):
    def forward(self, input):
        return input

class torchZDiversity(torchPupilDiversity):

    def __init__(self, 
                 z_list, 
                 nf=1.518, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
                 
        super(torchZDiversity, self).__init__(aperture_size, computation_size, N_pts)
        
        self.z_list = torch.reshape(torch.tensor(list(z_list)), (1,1,-1))
        self.N_zdiv = len(z_list)
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).type(torch.cfloat)
        zdiv = torch.exp(1j*2*torch.pi*self.nf*self.z_list*(1-ur**2)**(1/2))
        return zdiv.type(torch.cfloat)

class torchDerivativeDiversity(torchPupilDiversity):

    def __init__(self, 
                 m=2, 
                 ni=1.33, 
                 nf=1.518, 
                 aperture_size = 1., 
                 computation_size=4., 
                 N_pts=128):

        super(torchDerivativeDiversity, self).__init__(aperture_size, computation_size, N_pts)
        
        self.m = m
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        nr, nc = ur.shape
        ur = ur.type(torch.cfloat)
        pupil_array = torch.empty((nr,nc,self.m+1), dtype=torch.cfloat)
        for l in range(self.m+1):
            pupil_array[...,l] = (1j*2*torch.pi*self.ni*(1-(self.nf*ur/self.ni)**2)**(1/2))**l
        
        return pupil_array


class torchDDiversity(torchPupilDiversity):

    def __init__(self, 
                 diff_del_list, 
                 ni=1.33, 
                 nf=1.518, 
                 aperture_size = 1., 
                 computation_size=4., 
                 N_pts=128):

        super(torchZDiversity, self).__init__(aperture_size, computation_size, N_pts)
        
        self.diff_del_list = torch.reshape(torch.tensor(diff_del_list), (1,1,-1))
        self.N_ddiv = len(diff_del_list)
        self.ni = ni
        self.nf = nf  

    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = (ur[...,None]).type(torch.cfloat)
        return torch.exp(1j*2*torch.pi*self.diff_del_list*self.ni
                      *(1-(self.nf*ur/self.ni)**2)**(1/2)) 