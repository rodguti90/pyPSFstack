#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 08:11:32 2021

@author: rodrigo
"""

import numpy as np

class Pupil():
    
    def __init__(self, aperture_size, computation_size, N_pts):

        self.aperture_size = aperture_size 
        self.computation_size = computation_size
        self.N_pts = N_pts
        self.set_aperture()

    def step_fourier(self):
        return self.computation_size/self.N_pts
    
    def xy_mesh(self):
        u_vec = np.arange(-self.computation_size/2,
                        self.computation_size/2,
                        self.step_fourier())
        return np.meshgrid(u_vec,u_vec)
    
    def polar_mesh(self):
        ux, uy = self.xy_mesh()
        ur = np.sqrt(ux**2 + uy**2)
        uphi = np.arctan2(uy, ux)
        return ur, uphi

    def set_aperture(self):
        ur, _ = self.polar_mesh()
        self.aperture = np.empty((self.N_pts,self.N_pts,1,1), dtype=np.cfloat)
        self.aperture[...,0,0] = ur**2 <= self.aperture_size**2
           
    # def update_pupil_parameters(self, pupil_params):
    #     self.pupil_params = pupil_params
    #     self.generate_pupil_array()

    def generate_pupil_array(self):       
        raise NotImplementedError("Please Implement this method")




              