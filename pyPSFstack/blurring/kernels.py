import numpy as np
from scipy.special import jv, j1, spherical_jn
from math import factorial
from ..pupil import BlurringKernel

class BKSphere(BlurringKernel):
    def __init__(self, 
                 diff_del_list,
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):

        BlurringKernel.__init__(self, aperture_size, computation_size, N_pts)
        self.diff_del_list = diff_del_list
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        # ur = ur[...,None]   
        rad_d = self.nf * (self.radius**2 - self.diff_del_list**2)**(1/2)
        bk = rad_d * j1(2*np.pi*ur[...,None]*rad_d) / ur[...,None]
        origin = ur == 0
        bk[origin,:] = rad_d**2 * np.pi

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap


class BKSASphere(BlurringKernel):
    def __init__(self, 
                 m,
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):

        BlurringKernel.__init__(self, aperture_size, computation_size, N_pts)
        self.l_max = m//2
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        nr, nc = ur.shape
        ur = ur.astype(np.cfloat)
        bk = np.empty((nr,nc,self.l_max+1), dtype=np.cfloat) 
        
        origin = ur == 0
        for l in range(self.l_max+1):
            pref = (self.nf*self.radius)**(l+2) / (2**l * factorial(l) *self.nf**(2*l+1))
            bessel_term = spherical_jn(l+1, 2*np.pi*self.nf*self.radius*ur)/(2*np.pi*ur)**(l+1)
            bessel_term[origin] = 2**l * factorial(l) * self.nf * self.radius / factorial(2*l)
            bk[...,l] = pref * bessel_term

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap

