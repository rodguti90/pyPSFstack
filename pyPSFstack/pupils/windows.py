#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:34:02 2021

@author: rodrigo
"""
import numpy as np

from ..pupil import Pupil
from ..diversities.pola_diversities import jones_qwp

class NoPupil(Pupil):
    def __init__(sefl):
        pass 
    
    def get_pupil_array(self):
        return np.array([[1,0],[0,1]])

class SEO(Pupil):
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        self.c = c

    def get_pupil_array(self):
        ur, uphi = self.polar_mesh()
        return self.get_aperture() * jones_seo(ur, uphi, c=self.c)


class SEOQuarter(Pupil):
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi, theta = np.pi/4):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        self.c = c
        self.theta = theta

    def get_pupil_array(self):
        ur, uphi = self.polar_mesh()
        seo = jones_seo(ur, uphi, c=self.c)
        return self.get_aperture() * (jones_qwp(self.theta) @ seo)


def jones_seo(ur, uphi, c=1.24*np.pi):
    ny, nx = ur.shape
    jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
    jones_mat[...,0,0] = np.cos(c*ur/2) +1j*np.sin(c*ur/2)*np.cos(uphi)
    jones_mat[...,0,1] = -1j*np.sin(c*ur/2)*np.sin(uphi)
    jones_mat[...,1,0] = jones_mat[...,0,1]
    jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
    return jones_mat


