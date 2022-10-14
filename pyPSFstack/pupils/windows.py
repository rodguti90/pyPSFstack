#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:34:02 2021

@author: rodrigo
"""
import numpy as np

from ..pupil import BirefringentWindow, ScalarWindow
from ..diversities.pola_diversities import jones_qwp

# class NoPupil(Pupil):
#     def __init__(sefl):
#         pass 
    
#     def get_pupil_array(self):
#         return np.array([[1,0],[0,1]])

class Defocus(ScalarWindow):
    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, nf=1.518, delta_z=0
                 ):
        super(Defocus, self).__init__(aperture_size, computation_size, N_pts)
        
        self.nf = nf
        self.delta_z = delta_z
        
    def get_pupil_array(self):
        ur, _ = self.polar_mesh()
        ur = ur.astype(np.cfloat)
        aperture = self.get_aperture(dummy_ind=0)
        defocus = np.exp(1j*2*np.pi*self.nf*self.delta_z*(1-ur**2)**(1/2))
        return aperture * defocus


class SEO(BirefringentWindow):
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi, phi=0):
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        self.c = c
        self.phi = phi

    def get_pupil_array(self):
        ur, uphi = self.polar_mesh()
        return self.get_aperture() * jones_seo(ur, uphi, c=self.c, phi=self.phi)

    # def forward(self, input):
    #     return self.get_pupil_array() @ input

class SEOQuarter(BirefringentWindow):
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi, theta = np.pi/4):
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        self.c = c
        self.theta = theta

    def get_pupil_array(self):
        ur, uphi = self.polar_mesh()
        seo = jones_seo(ur, uphi, c=self.c)
        return self.get_aperture() * (jones_qwp(self.theta) @ seo)

class Qplate(BirefringentWindow):
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, q=1, alpha=0):
        BirefringentWindow.__init__(self, aperture_size, computation_size, N_pts)
        self.q = q
        self.alpha = alpha

    def get_pupil_array(self):
        _, uphi = self.polar_mesh()
        return self.get_aperture() * jones_qplate(uphi, self.q, self.alpha)


def jones_qplate(uphi, q, alpha):
    ny, nx = uphi.shape
    jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
    theta = q*uphi + alpha
    jones_mat[...,0,0] = 1j*np.cos(2*theta)
    jones_mat[...,0,1] = 1j*np.sin(2*theta)
    jones_mat[...,1,0] = -np.conj(jones_mat[...,0,1])
    jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
    return jones_mat

def jones_seo(ur, uphi, c=1.24*np.pi, phi=0):
    ny, nx = ur.shape
    jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
    jones_mat[...,0,0] = np.cos(c*ur/2) +1j*np.sin(c*ur/2)*np.cos(uphi-2*phi)
    jones_mat[...,0,1] = -1j*np.sin(c*ur/2)*np.sin(uphi-2*phi)
    jones_mat[...,1,0] = jones_mat[...,0,1]
    jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
    return jones_mat


