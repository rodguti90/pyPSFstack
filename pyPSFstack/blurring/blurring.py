import numpy as np
from math import factorial
from ..diversities.pupil_diversities import NoDiversity, DDiversity, DerivativeDiversity
from ..blurring.kernels import BKSphere, BKSASphere
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
        output = self._compute_even_intensity_der(input, orientation)
        output = self.bk.forward(output)
        return np.abs(output)
        # return output

    def _compute_even_intensity_der(self, input, orientation):
        
        if orientation == [0,0,0]:
            field_m = input
        else:
            field_m = input @ (np.array(orientation))
        
        in_sh = list(input.shape)
        l_max = self.m_max//2 +1
        int_m = np.zeros(in_sh[:2]+[l_max]+in_sh[3:], dtype=np.cfloat)

        for l in range(l_max):
            for k in range(2*l+1):
                int_m[:,:,l,...] += (factorial(2*l)/(factorial(k)*factorial(2*l-k))) * \
                    np.conj(field_m[:,:,k,...]) * field_m[:,:,2*l-k,...]
        
        if orientation == [0,0,0]:
            return np.sum(int_m, axis=(-2,-1))
        else:
            return np.sum(int_m, axis=-1)

