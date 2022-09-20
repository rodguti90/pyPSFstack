#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:35:07 2021

@author: rodrigo
"""
import numpy as np
from ..pupil import Pupil
from .aberration_functions import zernike_sequence, defocus_j

class UnitaryAberrations(Pupil):
    '''
    UnitaryAberrations defines a pupil composed of a general Jones matrix where its elements
    are expanded in terms of Zernike polynomials. 
    Note: The piston term controlling the overall amplutde of the mask is fixed to one 
    (if we want to optimize it we'll need to change it but it is redundant with photobleaching 
    amplitudes) and the piston and defocus terms of the scalar phase are ommitted.
    '''
    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, jmax_list=[15]*5, index_convention='fringe'):
        
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        
        assert len(jmax_list) == 5
        self.jmax_list = jmax_list
        self.pupil_params = np.zeros(np.sum(self.jmax_list)-2)
        self.pupil_params[0] = 1
        x, y = self.xy_mesh()
        self.zernike_seq = zernike_sequence(np.max(self.jmax_list), 
                                            index_convention, 
                                            x/self.aperture_size, 
                                            y/self.aperture_size)
        self.defocus_j = defocus_j(index_convention)
        self.generate_pupil_array()

    def generate_pupil_array(self): 
        # Computes the unitary decomposition matrix and scalar terms         
        #initialize the matrix term   
        self.Q = np.zeros((self.N_pts,self.N_pts,2,2), dtype=np.cfloat)
        self.qs = np.zeros((self.N_pts,self.N_pts,4))  
        cum_j = 0           
        #Compute the qs
        for k in range(4):
            temp = self.pupil_params[cum_j:cum_j+self.jmax_list[k]]
            self.qs[...,k] = np.sum(self.zernike_seq[...,:self.jmax_list[k]] * temp, axis=2)                   
            cum_j += self.jmax_list[k]   
        #Compute the matrix term    
        self.Q[...,0,0] = self.qs[...,0] + 1j * self.qs[...,3]
        self.Q[...,0,1] = self.qs[...,2] + 1j * self.qs[...,1]
        self.Q[...,1,0] = -np.conj(self.Q[...,0,1])
        self.Q[...,1,1] = np.conj(self.Q[...,0,0])
        #Compute the wavefront
        # the phase term needs to be computed separetly since it
        # omits specific zernikes (piston 0 and defocus 4)       
        temp = np.hstack((self.pupil_params[cum_j:cum_j+self.defocus_j-1], [0],
             self.pupil_params[cum_j+self.defocus_j-1:]))     
        self.W = np.sum(self.zernike_seq[...,1:self.jmax_list[4]] * temp, axis = 2)
        cum_j += self.jmax_list[1]-2 
        #Compute the scalar term
        self.Gamma = np.empty((self.N_pts,self.N_pts,1,1), dtype=np.cfloat)
        self.Gamma[...,0,0] = np.exp(1j * 2 * np.pi * self.W)
        self.pupil_array = self.Gamma * self.Q        

    def compute_parameters_gradient(self, grad_pupil):
        grad_Gamma =np.sum(np.conj(self.Q)*grad_pupil,axis=(-2,-1))
        grad_Q = np.conj(self.Gamma)*grad_pupil        
        grad_W = 2*np.pi*np.imag(grad_Gamma*np.conj(self.Gamma[...,0,0]))
        grad_qs = np.empty_like(self.qs)
        grad_qs[...,0] = np.real(grad_Q[...,0,0]+grad_Q[...,1,1])
        grad_qs[...,1] = np.imag(grad_Q[...,0,1]+grad_Q[...,1,0])
        grad_qs[...,2] = np.real(grad_Q[...,0,1]-grad_Q[...,1,0])
        grad_qs[...,3] = np.imag(grad_Q[...,0,0]-grad_Q[...,1,1])
        # compute Zernike coefs gradient
        self.grad_pupil_params = np.zeros_like(self.pupil_params)    
        cum_j=0    
        for k in range(4):
            self.grad_pupil_params[cum_j:cum_j+self.jmax_list[k]] = \
                np.sum(grad_qs[...,np.newaxis,k]*self.zernike_seq[...,:self.jmax_list[k]], axis=(0,1))
            cum_j += self.jmax_list[k]       
        temp = np.sum(grad_W[...,np.newaxis]
            *self.zernike_seq[:,:,1:self.jmax_list[4]], axis=(0,1))
        self.grad_pupil_params[cum_j:cum_j+self.defocus_j-1] = temp[:self.defocus_j-1]
        self.grad_pupil_params[cum_j+self.defocus_j-1:cum_j+self.jmax_list[1]-2] = \
            temp[self.defocus_j:self.jmax_list[4]]
         