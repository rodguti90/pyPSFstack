import torch

from ..pupil import torchBlurringKernel


class torchBK2DSphere(torchBlurringKernel):
    def __init__(self, 
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):

        super(torchBK2DSphere, self).__init__(aperture_size, computation_size, N_pts)
        
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        # ur = ur.type(torch.cfloat)
        
        bk = self.nf**2 * self.radius**3 * ja1(2*torch.pi*self.nf*self.radius*ur)
        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=0)

        return bk * ap

class torchBKSASphere(torchBlurringKernel):
    def __init__(self, 
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):

        super(torchBKSASphere, self).__init__(aperture_size, computation_size, N_pts)
        
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        nr, nc = ur.shape
        # ur = ur.type(torch.cfloat)
        bk = torch.empty((nr,nc,2), dtype=torch.cfloat) 
        
        bk[...,0] = self.nf**2 * self.radius**3 * ja1(2*torch.pi*self.nf*self.radius*ur)
        bk[...,1] = self.nf**2 * self.radius**5 *ja2(2*torch.pi*self.nf*self.radius*ur) / 2 

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap

def ja1(x):
    l_val = x > 0.1
    s_val = torch.logical_not(l_val)
    output = torch.zeros_like(x)
    xl = x[l_val]
    xs = x[s_val]
    output[l_val] = torch.sin(xl)/xl**3 - torch.cos(xl)/xl**2
    output[s_val] = 1 / 3 -  xs**2 / 30
    return output 


def ja2(x):
    l_val = x > 0.1
    s_val = torch.logical_not(l_val)
    output = torch.zeros_like(x)
    xl = x[l_val]
    xs = x[s_val]
    output[l_val] = (torch.sin(xl)*(3/xl**2-1) - 3*torch.cos(xl)/xl)/xl**3
    output[s_val] =  1/15 - xs**2 / 210
    return output 
