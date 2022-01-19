#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:34:02 2021

@author: rodrigo
"""
import numpy as np

from ..pupil import Pupil

class NoPupil(Pupil):
    def __init__(self):
        self.generate_pupil_array()

    def generate_pupil_array(self):
        self.pupil_array = np.array([[1,0],[0,1]])

class SEO(Pupil):
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        self.c = c

        self.generate_pupil_array()

    def generate_pupil_array(self):
        ur, uphi = self.polar_mesh()
        self.pupil_array = np.empty((self.N_pts,self.N_pts,2,2), dtype=np.cfloat)
        self.pupil_array[...,0,0] = np.cos(self.c*ur/2) -1j*np.sin(self.c*ur/2)*np.cos(uphi)
        self.pupil_array[...,0,1] = -1j*np.sin(self.c*ur/2)*np.sin(uphi)
        self.pupil_array[...,1,0] = self.pupil_array[...,0,1]
        self.pupil_array[...,1,1] = np.conj(self.pupil_array[...,0,0])
        self.pupil_array = self.aperture * self.pupil_array

class SEOQuarter(Pupil):
    def __init__(self, aperture_size = 1., computation_size=4., 
                 N_pts=128, c=1.24*np.pi):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        self.c = c

        self.generate_pupil_array()

    def generate_pupil_array(self):
        ur, uphi = self.polar_mesh()
        self.pupil_array = np.empty((self.N_pts,self.N_pts,2,2), dtype=np.cfloat)
        self.pupil_array[...,0,0] = (1+1j)*(np.cos(self.c*ur/2) + 1j*np.sin(self.c*ur/2)*np.exp(1j*uphi))/2
        self.pupil_array[...,0,1] = (1-1j)*(np.cos(self.c*ur/2) - 1j*np.sin(self.c*ur/2)*np.exp(1j*uphi))/2
        self.pupil_array[...,1,0] = -1j * np.conj(self.pupil_array[...,0,1])
        self.pupil_array[...,1,1] = 1j * np.conj(self.pupil_array[...,0,0])

class WindowPupil(Pupil):

    def __inti__(self):
        Pupil.__init__(self)
        
    def generate_pupil_array(self, window_type, **kwargs):
        self.set_aperture()
        if window_type == 'SEO':
            ur, uphi = self.polar_mesh()
            self.pupil_array = jones_seo(ur, uphi, **kwargs)
        elif window_type == 'SEO_quarter':
            ur, uphi = self.polar_mesh()
            self.pupil_array = jones_seo_quarter(ur, uphi, **kwargs)
        else:
            raise NotImplementedError("The chosen window_type is not implemented")
        self.pupil_array = self.pupil_array * self.aperture

def jones_seo(ur, uphi, c=1.24*np.pi):
    ny, nx = ur.shape
    jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
    jones_mat[...,0,0] = np.cos(c*ur/2) -1j*np.sin(c*ur/2)*np.cos(uphi)
    jones_mat[...,0,1] = -1j*np.sin(c*ur/2)*np.sin(uphi)
    jones_mat[...,1,0] = jones_mat[...,0,1]
    jones_mat[...,1,1] = np.conj(jones_mat[...,0,0])
    return jones_mat

def jones_seo_quarter(ur, uphi, c=1.24*np.pi):
    ny, nx = ur.shape
    jones_mat = np.empty((ny,nx,2,2), dtype=np.cfloat)
    jones_mat[...,0,0] = (1+1j)*(np.cos(c*ur/2) + 1j*np.sin(c*ur/2)*np.exp(1j*uphi))/2
    jones_mat[...,0,1] = (1-1j)*(np.cos(c*ur/2) - 1j*np.sin(c*ur/2)*np.exp(1j*uphi))/2
    jones_mat[...,1,0] = -1j * np.conj(jones_mat[...,0,1])
    jones_mat[...,1,1] = 1j * np.conj(jones_mat[...,0,0])
    return jones_mat

def jones_identity(nx, ny):
    jones_mat = np.zeros((ny,nx,2,2), dtype=np.cfloat)
    jones_mat[...,0,0] = 1
    jones_mat[...,1,1] = 1
    return jones_mat