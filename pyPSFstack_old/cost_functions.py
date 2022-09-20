import numpy as np
class CostFunction():

    def compute_cost(self):
        raise NotImplementedError("Please Implement this method")

    def compute_gradient(self):
        raise NotImplementedError("Please Implement this method")

class CostPoisson(CostFunction):

    def compute_cost(self, psf_stack, data_psf_stack, mask=1): 
        self.value= -sum(mask *(data_psf_stack *np.log(psf_stack/data_psf_stack) - psf_stack + data_psf_stack))

    def compute_gradient(self, psf_stack, data_psf_stack, mask=1):
        self.gradient = -mask *(data_psf_stack/psf_stack - 1)

class CostGaussian(CostFunction):

    def compute_cost(self, psf_stack, data_psf_stack, mask=1):       
        self.value = np.sum(mask*(psf_stack - data_psf_stack)**2)

    def compute_gradient(self, psf_stack, data_psf_stack, mask=1):
        self.gradient = 2*mask*(psf_stack - data_psf_stack)
