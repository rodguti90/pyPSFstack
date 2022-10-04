import torch as torch
from ..pupil import torchPupilDiversity


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