import torch
import torch.nn as nn

import numpy as np

from .functions import xy_mesh

def compute_green(ux, uy, ni, nf):
    
    N_pts = ux.shape[0] 
    ur2 = torch.empty((N_pts, N_pts), dtype=torch.cfloat)
    ur2[:,:] = ux**2 + uy**2
    # Compute the Phi coefficients which include the Fresnel coefs
    Phi1 = 2 * nf**2 * (1 - ur2)**(1/2) / \
        (nf * ni * (1 - nf**2 * ur2 / ni**2)**(1/2)+ ni**2 * (1 - ur2)**(1/2)) 
    Phi2 = (2  * nf * (1 - nf**2 * ur2 / ni**2)**(1/2)) / \
        (nf * (1 - nf**2 * ur2 /ni**2)**(1/2)+ ni * (1 - ur2)**(1/2))
    Phi3 = 2 * nf * (1 - ur2)**(1/2) / \
        (ni * (1 - nf**2 * ur2 / ni**2)**(1/2)+ nf * (1 - ur2)**(1/2))
    # The conservation of energy factor
    con_en = torch.empty((N_pts,N_pts,1,1), dtype=torch.cfloat)
    con_en[...,0,0] = (1 - ur2)**(1/4)

    green_mat = torch.empty((N_pts,N_pts,2,3), dtype=torch.cfloat)

    green_mat[...,0,0] = (ux**2 * (1- ur2)**(1/2) * Phi2 + uy**2 * Phi3)/ ur2  
    green_mat[...,0,1] = (ux * uy * ((1- ur2)**(1/2) * Phi2 - Phi3))/ ur2
    green_mat[...,0,2] = - ux * Phi1
    green_mat[...,1,1] = (uy**2 * (1- ur2)**(1/2) * Phi2 + ux**2 * Phi3)/ ur2
    green_mat[...,1,2] = - uy * Phi1
    green_mat[...,1,0] = green_mat[...,0,1]
    origin = ur2 == 0
    green_mat[origin,0,0] = Phi2[origin]
    green_mat[origin,1,1] = Phi3[origin]
    green_mat = green_mat / con_en
    green_mat = torch.nan_to_num(green_mat, nan=0.0, posinf=0.0, neginf=0.0)
    return green_mat

class DipoleInterfaceSource(nn.Module):

    def __init__(self, aperture_size=.99, computation_size=4., 
                 N_pts=128, ni=1.33, nf=1.518, delta=0.1
                 ):
        super(DipoleInterfaceSource, self).__init__()

        self.alpha = nn.Parameter(torch.tensor((nf/ni)**3, requires_grad=True, dtype=torch.float))
        self.delta =nn.Parameter(torch.tensor(delta, requires_grad=True, dtype=torch.float))
        
        self.ni = ni
        self.nf = nf
        
        step = computation_size/N_pts
        # Limit the pupil to the maximum region of one to avoid wasting memory
        ux, uy = xy_mesh(2, step)
        self.N_pupil = ux.shape[0]
        self.ur2 = torch.empty((self.N_pupil,self.N_pupil), dtype=torch.cfloat)
        self.ur2[...] = ux**2 + uy**2
        aperture = self.ur2 <= aperture_size**2
        self.green = aperture[...,None,None] * compute_green(ux, uy, ni, nf)

    def forward(self):

        saf_defocus = torch.empty((self.N_pupil,self.N_pupil,1,1), dtype=torch.cfloat)
        
        saf_defocus[...,0,0] =  torch.exp(1j*2*np.pi*self.nf*self.delta
                            *((self.ni/self.nf)*(1-self.ur2*(self.nf/self.ni)**2)**(1/2)
                            -self.alpha*(1-self.ur2)**(1/2)))   
        
        return self.green * saf_defocus