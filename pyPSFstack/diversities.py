#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:34:02 2021

@author: rodrigo
"""
import numpy as np
from .pupil import Pupil

class ZDiversity(Pupil):
    def __init__(self, z_list, nf=1.518, aperture_size = 1., computation_size=4., 
                 N_pts=128):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        
        self.z_list = np.reshape(np.array(z_list), (1,1,-1))
        self.N_zdiv = len(z_list)
        self.nf = nf
        self.aperture = self.aperture.reshape(self.aperture.shape[:-1])
        self.generate_pupil_array()

    def generate_pupil_array(self):
        ur = np.empty((self.N_pts,self.N_pts,1), dtype=np.cfloat)
        ur[:,:,0], _ = self.polar_mesh()
        self.pupil_array = self.aperture*np.exp(1j*2*np.pi*self.nf*self.z_list*(1-ur**2)**(1/2))

class PDiversity():
    def __init__(self, pol_analyzer, angle_list=[]):
        self.angle_list = angle_list
        self.generate_jones_list(pol_analyzer)

    def generate_jones_list(self, pol_analyzer):
        if pol_analyzer == 'quarter2pol':
            self.jones_list = quarter2pol(self.angle_list)    
        else:
            raise NotImplementedError("The chosen polarization analyzer is not implemented")
        self.N_pdiv = len(self.jones_list)

def quarter2pol(angle_list):
    n_ang = len(angle_list)
    quart2pol_analyzer = np.zeros((2*n_ang,2,2), dtype=np.cfloat)
    quart2pol_analyzer[:n_ang,0,0] = (np.cos(angle_list)**2 + 1j*np.sin(angle_list)**2)
    quart2pol_analyzer[:n_ang,0,1] = (1-1j)*np.sin(angle_list)*np.cos(angle_list)
    quart2pol_analyzer[n_ang:,1,0] = quart2pol_analyzer[:n_ang,0,1]
    quart2pol_analyzer[n_ang:,1,1] = (1j*np.cos(angle_list)**2 + np.sin(angle_list)**2)   
    return quart2pol_analyzer