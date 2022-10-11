import torch
import torch.nn as nn

import numpy as np

from ..pupil import torchSource


class torchDipoleInterfaceSource(torchSource):

    def __init__(self, aperture_size=1, computation_size=4., 
                 N_pts=128, ni=1.33, nf=1.518, delta=0.1, 
                 alpha=None, opt_delta=False, opt_alpha=False
                 ):
        super(torchDipoleInterfaceSource, self).__init__(aperture_size, computation_size, N_pts)

        # self.alpha = nn.Parameter(torch.tensor((nf/ni)**3, requires_grad=True, dtype=torch.float))
        if alpha is None:
            nr = nf/ni
            alpha = (140*nr + 42*nr**3 + 42*nr**5 + 15*nr**7)/(140 + 84*nr**2 + 15*nr**4)
        
        if opt_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha, requires_grad=True, dtype=torch.float))
        else:
            self.alpha = alpha
        
        if opt_delta:
            self.delta = nn.Parameter(torch.tensor(delta, requires_grad=True, dtype=torch.float))
        else:
            self.delta = torch.tensor(delta, dtype=torch.float)
        self.ni = ni
        self.nf = nf
        

    def get_pupil_array(self):

        ux, uy = self.xy_mesh()
        ur, _ = self.polar_mesh()
        saf_defocus = compute_SAF_defocus(ur.type(torch.cfloat), 
            self.ni, self.nf, self.delta, self.alpha) 
        green = compute_green(ux, uy, self.ni, self.nf)
        aperture = self.get_aperture()
        return aperture * saf_defocus * green  


def compute_SAF_defocus(ur, ni, nf, delta, alpha):
    N_pts = ur.shape[0]
    saf_defocus = torch.empty((N_pts,N_pts,1,1), dtype=torch.cfloat)
        
    saf_defocus[...,0,0] =  torch.exp(1j*2*np.pi*nf*delta
                        *((ni/nf)*(1-(ur*nf/ni)**2)**(1/2)
                        -alpha*(1-ur**2)**(1/2)))
    return saf_defocus

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
    con_en[con_en==0] = 1e-10

    green_mat = torch.empty((N_pts,N_pts,2,3), dtype=torch.cfloat)

    green_mat[...,0,0] = (ux**2 * (1- ur2)**(1/2) * Phi2 + uy**2 * Phi3)/ ur2  
    green_mat[...,0,1] = (ux * uy * ((1- ur2)**(1/2) * Phi2 - Phi3))/ ur2
    green_mat[...,0,2] = - ux * Phi1
    green_mat[...,1,1] = (uy**2 * (1- ur2)**(1/2) * Phi2 + ux**2 * Phi3)/ ur2
    green_mat[...,1,2] = - uy * Phi1
    
    ind_origin = ((ur2 == 0).nonzero())
    if len(ind_origin)>0:
        ind_origin = ind_origin[0]
        green_mat[ind_origin[0],ind_origin[1],0,0] = Phi2[ind_origin[0],ind_origin[1]]
        green_mat[ind_origin[0],ind_origin[1],1,1] = Phi3[ind_origin[0],ind_origin[1]]
        green_mat[ind_origin[0],ind_origin[1],0,1] = 0

    green_mat[...,1,0] = green_mat[...,0,1]
    green_mat = green_mat / con_en
    return green_mat

