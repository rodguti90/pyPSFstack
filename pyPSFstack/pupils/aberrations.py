#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:35:07 2021

@author: rodrigo
"""
import numpy as np
from ..pupil import BirefringentWindow, ScalarWindow
from .zernike_functions import zernike_sequence, defocus_j

class UnitaryZernike(BirefringentWindow):
    '''
    UnitaryZernike defines a pupil composed of a general Jones matrix where its elements
    are expanded in terms of Zernike polynomials. 
    Note: The piston term controlling the overall amplutde of the mask is fixed to one 
    (if we want to optimize it we'll need to change it but it is redundant with photobleaching 
    amplitudes) and the piston and defocus terms of the scalar phase are ommitted.
    '''
    def __init__(self, c_W, c_q, aperture_size=1., computation_size=4., 
                 N_pts=128, index_convention='standard'):
        
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        
        # assert len(jmax_list) == 5
        # self.jmax_list = jmax_list
        self.jmax_list = []
        for i in range(4):
            self.jmax_list += [len(c_q[i])]
        self.jmax_list += [len(c_W)+2]
        self.c_q = c_q
        self.c_W = c_W
        self.index_convention = index_convention
        
        self.defocus_j = defocus_j(index_convention)

    def get_pupil_array(self): 

        x, y = self.xy_mesh()
        N_pts = x.shape[0]
        zernike_seq = zernike_sequence(np.max(self.jmax_list), 
                                        self.index_convention, 
                                        x/self.aperture_size, 
                                        y/self.aperture_size)
        # Computes the unitary decomposition matrix and scalar terms         
        #initialize the matrix term   
        Q = np.zeros((N_pts,N_pts,2,2), dtype=np.cfloat)
        qs = np.zeros((N_pts,N_pts,4))   
        #Compute the qs
        for k in range(4):
            qs[...,k] = np.sum(zernike_seq[...,:self.jmax_list[k]] 
                * self.c_q[k], axis=2)        
        #Compute the matrix term    
        Q[...,0,0] = qs[...,0] + 1j * qs[...,3]
        Q[...,0,1] = qs[...,2] + 1j * qs[...,1]
        Q[...,1,0] = -np.conj(Q[...,0,1])
        Q[...,1,1] = np.conj(Q[...,0,0])
        #Compute the wavefront
        # the phase term needs to be computed separetly since it
        # omits specific zernikes (piston 0 and defocus 4)       
        temp = np.hstack((self.c_W[:self.defocus_j-1], [0],
             self.c_W[self.defocus_j-1:]))     
        W = np.sum(zernike_seq[...,1:self.jmax_list[4]] * temp, axis = 2)
        
        #Compute the scalar term
        Gamma = np.empty((N_pts,N_pts,1,1), dtype=np.cfloat)
        Gamma[...,0,0] = np.exp(1j * 2 * np.pi * W)
        
        return Gamma * Q        



class ScalarZernike(ScalarWindow):
    '''
    UnitaryAberrations defines a pupil composed of a general Jones matrix where its elements
    are expanded in terms of Zernike polynomials. 
    Note: The piston term controlling the overall amplutde of the mask is fixed to one 
    (if we want to optimize it we'll need to change it but it is redundant with photobleaching 
    amplitudes) and the piston and defocus terms of the scalar phase are ommitted.
    '''
    def __init__(self, c_A, c_W, aperture_size=1., computation_size=4., 
                 N_pts=128, index_convention='standard'):
        
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)

        self.jmax_list = [len(c_A), len(c_W)+1]
        
        self.c_A = c_A
        self.c_W = c_W
        self.index_convention = index_convention
        
        # self.defocus_j = defocus_j(index_convention)

    def get_pupil_array(self): 

        x, y = self.xy_mesh()
        zernike_seq = zernike_sequence(np.max(self.jmax_list), 
                                        self.index_convention, 
                                        x/self.aperture_size, 
                                        y/self.aperture_size)
        
        Amp = np.sum(zernike_seq[...,:self.jmax_list[0]] 
                * self.c_A, axis=2)        
        #Compute the wavefront          
        W = np.sum(zernike_seq[...,1:self.jmax_list[1]] * self.c_W, axis = 2)
       
        
        return Amp *  np.exp(1j * 2 * np.pi * W)       
    


class ApodizedUnitary(BirefringentWindow):
    '''
    UnitaryAberrations defines a pupil composed of a general Jones matrix where its elements
    are expanded in terms of Zernike polynomials. 
    Note: The piston term controlling the overall amplutde of the mask is fixed to one 
    (if we want to optimize it we'll need to change it but it is redundant with photobleaching 
    amplitudes) and the piston and defocus terms of the scalar phase are ommitted.
    '''
    def __init__(self, c_A, c_W, c_q, aperture_size=1., computation_size=4., 
                 N_pts=128, index_convention='standard'):
        
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        
        # assert len(jmax_list) == 5
        # self.jmax_list = jmax_list
        self.jmax_list = []
        for i in range(4):
            self.jmax_list += [len(c_q[i])]
        self.jmax_list += [len(c_W)+2]
        self.c_q = c_q
        self.c_W = c_W
        self.c_A = c_A

        self.jmax_list += [len(c_A)]
        self.index_convention = index_convention
        
        self.defocus_j = defocus_j(index_convention)

    def get_pupil_array(self): 

        x, y = self.xy_mesh()
        N_pts = x.shape[0]
        zernike_seq = zernike_sequence(np.max(self.jmax_list), 
                                        self.index_convention, 
                                        x/self.aperture_size, 
                                        y/self.aperture_size)
        # Computes the unitary decomposition matrix and scalar terms         
        #initialize the matrix term   
        qs = np.zeros((N_pts,N_pts,4))   
        for k in range(4):
            qs[...,k] = np.sum(zernike_seq[...,:self.jmax_list[k]] 
                * self.c_q[k], axis=2)        
        #Compute the matrix term    
        Q = np.zeros((N_pts,N_pts,2,2), dtype=np.cfloat)
        Q[...,0,0] = qs[...,0] + 1j * qs[...,3]
        Q[...,0,1] = qs[...,2] + 1j * qs[...,1]
        Q[...,1,0] = -np.conj(Q[...,0,1])
        Q[...,1,1] = np.conj(Q[...,0,0])
        normQ = np.sum(np.abs(qs)**2, axis=-1)**(1/2)
        
        (qs[0]**2+qs[1]**2+qs[2]**2+qs[3]**2)**(1/2)
        normQ[normQ==0] = 1

        temp = np.hstack((self.c_W[:self.defocus_j-1], [0],
             self.c_W[self.defocus_j-1:]))     
        W = np.sum(zernike_seq[...,1:self.jmax_list[-2]] * temp, axis = 2)
        Amp = np.sum(zernike_seq[...,:self.jmax_list[-1]] * self.c_A, axis = 2)
        #Compute the scalar term
        Gamma = np.empty((N_pts,N_pts,1,1), dtype=np.cfloat)
        Gamma[...,0,0] = Amp * np.exp(1j * 2 * np.pi * W)/normQ
        
        return Gamma * Q 