import numpy as np
from scipy.special import jv 
from math import factorial
from .pupil import BlurringKernel
from .diversities.pupil_diversities import NoDiversity, DDiversity, DerivativeDiversity
# from .core import PSFStack


class Blurring():
    def __init__(self):
                 
        self.diversity = NoDiversity()  
    
    def compute_blurred_psfs(self, input):
        raise NotImplementedError("Please Implement this method")

    def _compute_intensity(self, input, orientation):
        if orientation == [0,0,0]:
            return np.sum(np.abs(input)**2, axis=(-2,-1))
        else:
            field = input @ (np.array(orientation))
            return np.sum(np.abs(field)**2, axis=-1)

class NoBlurring(Blurring):

    def __init__(self):
        self.diversity = NoDiversity()

    def compute_blurred_psfs(self, input, orientation):
        return self._compute_intensity(input, orientation)


class ExactBlurring(Blurring):

    def __init__(self,
                 radius=0.,
                 diff_del_list=[],
                 emission='sphere',
                 ni=1.33,
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 ):

        Blurring.__init__(self)
        
        self.diversity = DDiversity(diff_del_list, 
                                    ni=ni, 
                                    nf=nf, 
                                    aperture_size=aperture_size, 
                                    computation_size=computation_size, 
                                    N_pts=N_pts)

        if emission == 'sphere':
            self.bk = BKSphere(diff_del_list,
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts)
        

    def compute_blurred_psfs(self, input, orientation):
        output = self._compute_intensity(input, orientation)
        output = self.bk.forward(output)
        return np.abs(output)

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
        bk = rad_d * jv(1, 2*np.pi*ur[...,None]*rad_d) / ur[...,None]
        origin = ur == 0
        bk[origin,:] = rad_d**2 * np.pi

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap


class SABlurring(Blurring):

    def __init__(self,
                 radius=0.,
                 m=2,
                 emission='sphere',
                 ni=1.33,
                 nf=1.518,          
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128,
                 ):

        Blurring.__init__(self)
        
        self.diversity = DerivativeDiversity(m=m, 
                                    ni=ni, 
                                    nf=nf, 
                                    aperture_size=aperture_size, 
                                    computation_size=computation_size, 
                                    N_pts=N_pts)
        self.m_max = m
        if emission == 'sphere':
            self.bk = BKSASphere(m,
                               radius,
                               nf,
                               aperture_size=2*aperture_size,
                               computation_size=computation_size,
                               N_pts=N_pts)
        

    def compute_blurred_psfs(self, input, orientation):
        output = self._compute_intensity_der(input, orientation)
        output = self.bk.forward(output)
        return np.abs(output)

    def _compute_intensity_der(self, input, orientation):
        int_m = np.zeros_like(input)
        if orientation != [0,0,0]:
            input @ (np.array(orientation))
            
        for m in range(self.m_max+1):
            for l in range(m):
                int_m[:,:,m,...] += (factorial(m)/(factorial(l)*factorial(m-l))) * \
                    np.conj(input[:,:,l,...]) * np.conj(input[:,:,m-l,...])
        if orientation == [0,0,0]:
            return np.sum(int_m, axis=(-2,-1))
        else:
            field = int_m @ (np.array(orientation))
            return np.sum(int_m, axis=-1)


class BKSASphere(BlurringKernel):
    def __init__(self, 
                 m,
                 radius,
                 nf,
                 aperture_size=2.,
                 computation_size=4,
                 N_pts=128):

        BlurringKernel.__init__(self, aperture_size, computation_size, N_pts)
        self.N_derdiv = m+1
        self.nf = nf
        self.radius = radius

    def get_pupil_array(self):
        ur, _ = self.polar_mesh(umax=self.computation_size/2)
        # ur = ur[...,None]   
        rad_d = self.nf * (self.radius**2 - self.diff_del_list**2)**(1/2)
        bk = rad_d * jv(1, 2*np.pi*ur[...,None]*rad_d) / ur[...,None]
        origin = ur == 0
        bk[origin,:] = rad_d**2 * np.pi

        ap = self.get_aperture(umax=self.computation_size/2, dummy_ind=1)
        return bk * ap

