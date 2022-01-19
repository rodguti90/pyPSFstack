#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 18:31:35 2021

@author: rodrigo
"""
import numpy as np
from ..pupil import Pupil


class DipoleInterfaceSource(Pupil):
    def __init__(self, aperture_size=1., computation_size=4., 
                 N_pts=128, ni=1.33, nf=1.518, delta=0.1):
        Pupil.__init__(self, aperture_size, computation_size, N_pts)
        
        self.ni = ni
        self.nf = nf
        alpha = (self.nf/self.ni)**3
        self.pupil_params = [alpha, delta]        
        
        self.compute_green()
        self.generate_pupil_array()
    
    def generate_pupil_array(self):     
        self.compute_SAF_defocus()   
        self.pupil_array = self.saf_defocus * self.green_mat        

    def compute_SAF_defocus(self):
        self.saf_defocus = np.empty((self.N_pts,self.N_pts,1,1), dtype=np.cfloat)
        ur = np.empty((self.N_pts,self.N_pts), dtype=np.cfloat)
        ur[:,:], _ = self.polar_mesh()
        self.saf_defocus[...,0,0] =  np.exp(1j*2*np.pi*self.nf*self.pupil_params[1]
                            *((self.ni/self.nf)*(1-(self.nf*ur/self.ni)**2)**(1/2)
                            -self.pupil_params[0]*(1-ur**2)**(1/2)))   
        self.saf_defocus = self.aperture * self.saf_defocus

    def compute_green(self):
        """
        green_BFP calculates the Green tensor at the BFP taking into account 
        the supercritical radiation 
        The distance to the coverslip needs to be specified as a positive number
        since the origin is placed at the glass/medium boundary. In the same vein
        the focal plane is located at the same boundary. In order to consider
        another focal plane we need to multiply the Green tensor by the
        corresponding phase factor where a positive distance means that the
        focal plane is in the immersion medium. Conversly if it is negative then
        the focal plane is in the suspension medium (where the particles are).
        """
        np.seterr(divide='ignore', invalid='ignore')
        ni = self.ni
        nf = self.nf
        ux, uy = self.xy_mesh()
        ur2 = np.empty((self.N_pts,self.N_pts),dtype=np.cfloat)
        ur2[:,:] = ux**2 + uy**2

        # Compute the Phi coefficients which include the Fresnel coefs
        Phi1 = 2 * nf**2 * (1 - ur2)**(1/2) / \
            (nf * ni * (1 - nf**2 * ur2 / ni**2)**(1/2)+ ni**2 * (1 - ur2)**(1/2)) 
        Phi2 = (2  * nf * (1 - nf**2 * ur2 / ni**2)**(1/2)) / \
            (nf * (1 - nf**2 * ur2 /ni**2)**(1/2)+ ni * (1 - ur2)**(1/2))
        Phi3 = 2 * nf * (1 - ur2)**(1/2) / \
            (ni * (1 - nf**2 * ur2 / ni**2)**(1/2)+ nf * (1 - ur2)**(1/2))
        # The conservation of energy factor
        con_en = np.empty((self.N_pts,self.N_pts,1,1), dtype=np.cfloat)
        con_en[...,0,0] = (1 - ur2)**(1/4)

        self.green_mat = np.empty((self.N_pts,self.N_pts,2,3), dtype=np.cfloat)

        self.green_mat[...,0,0] = (ux**2 * (1- ur2)**(1/2) * Phi2 + uy**2 * Phi3)/ ur2  
        self.green_mat[...,0,1] = (ux * uy * ((1- ur2)**(1/2) * Phi2 - Phi3))/ ur2
        self.green_mat[...,0,2] = - ux * Phi1
        self.green_mat[...,1,1] = (uy**2 * (1- ur2)**(1/2) * Phi2 + ux**2 * Phi3)/ ur2
        self.green_mat[...,1,2] = - uy * Phi1
        self.green_mat[...,1,0] = self.green_mat[...,0,1]
        origin = ur2 == 0
        self.green_mat[origin,0,0] = Phi2[origin]
        self.green_mat[origin,1,1] = Phi3[origin]
        self.green_mat = self.aperture * self.green_mat / con_en
        self.green_mat = np.nan_to_num(self.green_mat, nan=0.0, posinf=0.0, neginf=0.0)
        
    def compute_parameters_gradient(self, grad_pupil):
        ur = np.empty((self.N_pts,self.N_pts), dtype=np.cfloat)
        ur[:,:], _ = self.polar_mesh()
        self.grad_saf_defocus = np.sum(grad_pupil * np.conj(self.green_mat), axis=(-2,-1))
        self.grad_exp_arg = np.conj(self.saf_defocus) * self.grad_saf_defocus
        self.grad_delta = np.real(-1j*2*np.pi*self.nf
            *np.sum(self.grad_exp_arg*np.conj((self.ni/self.nf)*(1-(self.nf*ur/self.ni)**2)**(1/2)
                            -self.pupil_params[0]*(1-ur**2)**(1/2))))
        self.grad_alpha = -2*np.pi*self.nf*np.sum(np.imag(self.grad_exp_arg)*(1-ur**2)**(1/2))
        self.grad_pupil_params = [self.grad_alpha, self.grad_delta]