import torch
from math import factorial


from ..pupil import torchBlurringKernel


class torchBKSASphere(torchBlurringKernel):
    def __init__(self, 
                 m,
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):

        super(torchBKSASphere, self).__init__(aperture_size, computation_size, N_pts)
        self.l_max = m//2
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        nr, nc = ur.shape
        ur = ur.type(torch.cfloat)
        bk = torch.empty((nr,nc,self.l_max+1), dtype=torch.cfloat) 
        
        origin = ur == 0
        for l in range(self.l_max+1):
            pref = (self.nf*self.radius)**(l+2) / (2**l * factorial(l) *self.nf**(2*l+1))
            bessel_term = spherical_jn(l+1, 2*torch.pi*self.nf*self.radius*ur)/(2*torch.pi*ur)**(l+1)
            bessel_term[origin] = 2**l * factorial(l) * self.nf * self.radius / factorial(2*l)
            bk[...,l] = pref * bessel_term

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap

def spherical_j1():
    pass

def spherical_j2():
    pass