"""
Created on Thu Dec 16 08:57:52 2021

@author: rodrigo
"""


from pyPSFstack_old.pupils.windows import NoPupil

import numpy as np
from scipy.optimize import minimize
from skimage.morphology import erosion, dilation

def outer_pixels(stack):
    shape_stack = list(stack.shape)
    [NX, NY] = np.meshgrid(np.arange(shape_stack[0]),
                           np.arange(shape_stack[1]))
    outer_pix = ((NX - (shape_stack[1]-1)/2)**2
                 + (NY - (shape_stack[0]-1)/2)**2
                 > ((np.min(shape_stack[:2])-1)/2)**2)

    return outer_pix.reshape(shape_stack[:2]+[1]*len(shape_stack[2:]))

def denoise_stack(stack):
    outer_pix = outer_pixels(stack)
    bckgd = estimate_background(stack, mask=outer_pix)
    std = np.std(stack, axis=(0,1), where=outer_pix)
    denoised_stack = stack - bckgd
    threshold_mask = denoised_stack > std
    threshold_mask = erosion(threshold_mask)
    threshold_mask = dilation(threshold_mask)
    denoised_stack = denoised_stack * threshold_mask
    denoised_stack[denoised_stack<0] = 0
    return denoised_stack, bckgd

def estimate_background(stack,mask=None):
    """
    ESTIMATE_BCKGD estimates the value of the background
    illumination and its standard deviation on the images by 
    computing the mean value on the pixels outside a circle of 
    radius NX/2         
    """  
    background = np.mean(stack, axis=(0,1), where=mask)
    return background
   
def estimate_photobleach_background(data, model=None):
    data_stack, bckgd = denoise_stack(data)
    amp_data = np.sum(data_stack, axis=(0,1))
    amp_data = amp_data / amp_data[0,0]
    if model is not None:
        if model.shape[0] != data.shape[0]:
            model = trim_stack(model, data.shape[0])
        model_stack, _ = denoise_stack(model)
        amp_model = np.sum(model_stack, axis=(0,1))
        amp_model = amp_model / amp_model[0,0]
    else:
        model_stack = 1
        amp_model = 1
    photobleach_amplitudes = amp_data / amp_model
    photobleach_amplitudes[photobleach_amplitudes>1]=1
    scale = np.sum(photobleach_amplitudes*model_stack) \
        / np.sum(data_stack)
    return photobleach_amplitudes, scale, scale*bckgd

def trim_stack(stack, N_new):
    shape_stack = stack.shape
    N_pts = shape_stack[0]
    trimmed_stack = stack[(N_pts-N_new)//2:(N_pts+N_new)//2,
                          (N_pts-N_new)//2:(N_pts+N_new)//2]
    return trimmed_stack

def zeropad_stack(stack, N_new):
    N_old = stack.shape[0]
    pad_width = [(N_new-N_old)//2, int(np.ceil((N_new-N_old)/2))]
    padded_stack = np.pad(stack, [pad_width]*2+[[0,0]]*(stack.ndim-2))
    return padded_stack

def dag(array):
    return np.conj(np.swapaxes(array,-2,-1))


class PSFStack():
    def __init__(self, pupils=[NoPupil()], zdiversity=None, pdiversity=None):
        self.pupils = pupils
        self.N_pupils = len(self.pupils)
        self.N_pts = self.pupils[0].N_pts
        self.zdiversity = zdiversity
        self.N_zdiv = zdiversity.N_zdiv
        self.pdiversity = pdiversity
        self.N_pdiv = pdiversity.N_pdiv
        self.optimize_photobleach = True
        self.optimize_background = True
        self.cost_evol =[]

    def _compute_compound_pupils(self):
        self.compound_pupils = [self.pupils[0].pupil_array]
        for ind in range(self.N_pupils-1):
            self.compound_pupils += [self.pupils[ind+1].pupil_array 
                @ self.compound_pupils[ind]]
        
    def _compute_zdiv_pupil(self):
        self.zdiv_pupil = self.compound_pupils[-1][...,np.newaxis,:,:] \
            * self.zdiversity.pupil_array[...,np.newaxis,np.newaxis]

    def _propagate_image_plane(self):    
        self.zdiv_psf = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(self.zdiv_pupil, axes=(0,1)), 
                axes=(0,1)), 
            axes=(0,1))/self.N_pts

    def _compute_pdiv_psf_fields(self):
        self.field_psf_stack = self.pdiversity.jones_list @ self.zdiv_psf[...,np.newaxis,:,:]

    def _incoherent_sum(self):
        self.psf_stack = np.sum(np.abs(self.field_psf_stack)**2,axis=(-2,-1))

    def compute_psf_stack(self):
        self._compute_compound_pupils()
        self._compute_zdiv_pupil()
        self._propagate_image_plane()
        self._compute_pdiv_psf_fields()
        self._incoherent_sum()

    def model_experimental_stack(self, bckgd_photons=20, N_photons=200, N_pts=None):
        rng = np.random.default_rng()
        max_value = np.max(self.psf_stack)
        if N_pts is not None:
            stack = trim_stack(self.psf_stack, N_pts)/max_value
        else:
            stack = self.psf_stack/max_value
        stack = np.round(N_photons*stack + bckgd_photons)
        stack = rng.poisson(stack)
        return stack

    def _photobleach_background(self):
        self.estimate_psf_stack = self.photobleach_amplitudes * self.psf_stack \
            + self.background
    
    def set_data_mask(self):
        self.data_mask = np.zeros((self.N_pts,self.N_pts,1,1))
        self.data_mask[self.data_psf_stack[...,0,0]!=0]=1

    def set_data_psf_stack(self, data_psf_stack):       
        self.photobleach_amplitudes, self.data_scale_factor, self.background = \
            estimate_photobleach_background(data_psf_stack, model=self.psf_stack)
        self.N_data = data_psf_stack.shape[0]
        self.data_psf_stack = zeropad_stack(self.data_scale_factor*data_psf_stack, self.N_pts)
        self.set_data_mask()

    def set_cost_function(self, cost):
        self.cost = cost

    def update_optimization_parameters(self, optimization_params):
        params_counter = 0
        for ind in range(self.N_pupils):
            if self.pupils[ind].optimization_on:
                pupil_params = optimization_params[params_counter
                        :params_counter+self.pupils[ind].N_pupil_params]
                params_counter += self.pupils[ind].N_pupil_params            
                self.pupils[ind].update_pupil_parameters(pupil_params)
        if self.optimize_photobleach:
            self.photobleach_amplitudes = np.reshape(optimization_params[params_counter
                :params_counter+self.N_zdiv*self.N_pdiv], (self.N_zdiv,self.N_pdiv))
            params_counter += self.N_zdiv*self.N_pdiv
        if self.optimize_background:
            self.background = np.reshape(optimization_params[params_counter
                :params_counter+self.N_zdiv*self.N_pdiv], (self.N_zdiv,self.N_pdiv))
            params_counter += self.N_zdiv*self.N_pdiv

    def cost_computation(self):       
        self.compute_psf_stack()        
        self._photobleach_background()
        self.cost.compute_cost(self.estimate_psf_stack, 
                                self.data_psf_stack, 
                                mask=self.data_mask)

    def gradient_computation(self):
        self.cost.compute_gradient(self.estimate_psf_stack, 
                                    self.data_psf_stack, 
                                    mask=self.data_mask)
        if self.optimize_background:
            self.grad_background = np.sum(self.cost.gradient, 
                axis=(0,1)).ravel()
        else:
            self.grad_background = []
        if self.optimize_photobleach:
            self.grad_photobleach = np.sum(self.cost.gradient*self.psf_stack, 
                axis=(0,1)).ravel()
        else:
            self.grad_photobleach = []
        self.grad_psf_stack = self.photobleach_amplitudes*self.cost.gradient
        self.backprop_psf_stack()
        for ind in range(self.N_pupils):
            if self.pupils[ind].optimization_on:    
                self.pupils[ind].compute_parameters_gradient(self.grad_pupils[ind])
           
    def backprop_psf_stack(self):
        self._compute_grad_field_psf_stack()
        self._compute_grad_zdiv_psf() 
        self._compute_grad_zdiv_pupil()
        self._compute_grad_compound_pupils()
        self._compute_grad_pupils() 

    def _compute_grad_field_psf_stack(self):
        self.grad_field_psf_stack = (2*self.grad_psf_stack[...,np.newaxis,np.newaxis]
                                    *self.field_psf_stack)
    
    def _compute_grad_zdiv_psf(self):
        self.grad_zdiv_psf = np.sum(dag(self.pdiversity.jones_list)
            @self.grad_field_psf_stack, axis=3)

    def _compute_grad_zdiv_pupil(self):
        self.grad_zdiv_pupil = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(self.grad_zdiv_psf, axes=(0,1)), 
                axes=(0,1)), 
            axes=(0,1))*self.N_pts
            
    def _compute_grad_compound_pupils(self):
        self.grad_compound_pupils = [np.sum(self.grad_zdiv_pupil
            *np.conj(self.zdiversity.pupil_array[...,np.newaxis,np.newaxis]), axis=2)]           
        for ind in range(self.N_pupils-1):
            self.grad_compound_pupils = [dag(self.pupils[-ind-1].pupil_array)
                @ self.grad_compound_pupils[-ind-1]] \
                    + self.grad_compound_pupils
    
    def _compute_grad_pupils(self):
        self.grad_pupils = [self.grad_compound_pupils[0]]
        for ind in range(self.N_pupils-1):
            self.grad_pupils += [self.grad_compound_pupils[ind+1]
                @ dag(self.compound_pupils[ind])]

    def optimize_parameters(self, maxiter=50):
        initial_parameters= self._get_optimization_parameters()
        
        def obj_fun(optimization_params):
            self.update_optimization_parameters(optimization_params)
            self.cost_computation()
            self.gradient_computation()
            grads = []
            for ind in range(self.N_pupils):
                grads = np.hstack((grads,self.pupils[ind].grad_pupil_params))
            grads = np.hstack((grads,self.grad_photobleach))
            grads = np.hstack((grads,self.grad_background))
            self.cost_evol += [self.cost.value]
            return self.cost.value, grads
        options = {}
        options['maxiter'] = maxiter
        opt_res = minimize(obj_fun, initial_parameters, jac=True, options=options)
        self.update_optimization_parameters(opt_res.x)
        return opt_res
        
    def _get_optimization_parameters(self):
        initial_parameters=[]
        for ind in range(self.N_pupils):
            if self.pupils[ind].optimization_on:
                initial_parameters = np.hstack((initial_parameters,self.pupils[ind].pupil_params))
        if self.optimize_photobleach:
            initial_parameters = np.hstack((initial_parameters,
                self.photobleach_amplitudes.ravel()))
        if self.optimize_background:
            initial_parameters = np.hstack((initial_parameters,
                self.background.ravel()))
        return initial_parameters


class Microscope():
    '''
    Defines all the parameters of the microscope used in the experiments.

    This is used to compute all the parameters recquired for the propagatror
    from those of the experimental system.
    '''
    def __init__(self, wavelength=525, distance_coverslip=100, nf=1.518, NA=1.49, magnification=100, cam_pixel_size = 6500):
        self.cam_pixel_size = cam_pixel_size


