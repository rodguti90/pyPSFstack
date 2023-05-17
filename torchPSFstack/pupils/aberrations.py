import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from ..pupil import torchBirefringentWindow, torchScalarWindow
from .zernike_functions import zernike_sequence, defocus_j

class torchUnitaryZernike(torchBirefringentWindow):

    def __init__(self, c_W=None, c_q=None, aperture_size=.99, computation_size=4., 
                 N_pts=128, jmax_list=[15]*5, index_convention='standard'
                 ):
        super(torchUnitaryZernike, self).__init__(aperture_size, computation_size, N_pts)
        
        self.jmax=jmax_list
        if c_q is None:
            tempq = torch.zeros((jmax_list[0]), dtype=torch.float)
            tempq[0] = 1
            cq_list = [nn.Parameter(tempq, requires_grad=True)]
            for i in range(1,4):
                cq_list += [nn.Parameter(torch.zeros(jmax_list[i], requires_grad=True, dtype=torch.float))]
        else:
            cq_list=[]
            for i in range(0,4):
                cq_list += [torch.nn.Parameter(torch.from_numpy(c_q[i]).type(torch.float), requires_grad=True)]
                self.jmax[i] = len(c_q[i])
        self.c_q = nn.ParameterList(cq_list)
        
        if c_W is None:
            self.c_W = nn.Parameter(torch.zeros(jmax_list[4]-2, requires_grad=True, dtype=torch.float))
        else:
            self.c_W = torch.nn.Parameter(torch.from_numpy(c_W).type(torch.float), requires_grad=True)
            self.jmax[-1]=len(c_W)+2
        self.index_convention = index_convention                  
        self.defocus_j = defocus_j(index_convention)
    
    def get_pupil_array(self):
        ux, uy = self.xy_mesh()
        N_pupil = ux.shape[0]
        zernike_seq = zernike_sequence(np.max(self.jmax), 
                                        self.index_convention, 
                                        ux/self.aperture_size, 
                                        uy/self.aperture_size)

        qs = torch.zeros((4,N_pupil,N_pupil),  dtype=torch.cfloat)
        for q_ind in range(4):
            qs[q_ind] = torch.sum(self.c_q[q_ind]*zernike_seq[...,:self.jmax[q_ind]],-1)
        Q = torch.zeros((N_pupil,N_pupil,2,2), dtype=torch.cfloat)
        Q[...,0,0] = qs[0] + 1j*qs[3]
        Q[...,0,1] = qs[2] + 1j*qs[1]
        Q[...,1,0] = -qs[2] + 1j*qs[1]
        Q[...,1,1] = qs[0] - 1j*qs[3]

        W = torch.sum(zernike_seq[...,1:self.defocus_j]*self.c_W[:self.defocus_j-1],-1)\
            + torch.sum(zernike_seq[...,self.defocus_j+1:self.jmax[4]]*
                        self.c_W[self.defocus_j-1:],-1)
        Gamma = self.get_aperture(dummy_ind=0)*torch.exp(1j*2*np.pi*W)

        return ( Gamma[...,None,None] * Q)



class torchScalarZernike(torchScalarWindow):

    def __init__(self, aperture_size=.99, computation_size=4., 
                 N_pts=128, jmax_list=[15]*2, index_convention='standard'
                 ):
        super(torchScalarZernike, self).__init__(aperture_size, computation_size, N_pts)
        
        tempA = torch.zeros((jmax_list[0]), dtype=torch.float)
        tempA[0] = 1
        self.c_A = nn.Parameter(tempA, requires_grad=True)

        self.c_W = nn.Parameter(torch.zeros(jmax_list[1]-1, requires_grad=True, dtype=torch.float))
        
        self.jmax=jmax_list
        self.index_convention = index_convention                  
        self.defocus_j = defocus_j(index_convention)
    
    def get_pupil_array(self):
        ux, uy = self.xy_mesh()
        zernike_seq = zernike_sequence(np.max(self.jmax), 
                                        self.index_convention, 
                                        ux/self.aperture_size, 
                                        uy/self.aperture_size)

        Amp = torch.sum(self.c_A*zernike_seq[...,:self.jmax[0]], -1)
        
        W = torch.sum(zernike_seq[...,1:self.jmax[1]]*self.c_W, -1)

        return self.get_aperture(dummy_ind=0)*Amp*torch.exp(1j*2*np.pi*W)


class torchUnitaryPixels(torchBirefringentWindow):

    def __init__(self, aperture_size=.99, computation_size=4., N_pts=128):
        super(torchUnitaryPixels, self).__init__(aperture_size, computation_size, N_pts)
        
        aperture = self.get_aperture(dummy_ind=0)
        N_pupil = aperture.shape[0]

        self.W = nn.Parameter(torch.zeros([N_pupil,N_pupil], requires_grad=True, dtype=torch.float))
        tempq = torch.zeros([4,N_pupil,N_pupil], dtype=torch.float)
        tempq[0] = aperture
        self.qs = nn.Parameter(tempq, requires_grad=True)
    
    def get_pupil_array(self):
        
        aperture = self.get_aperture(dummy_ind=0)
        N_pupil = aperture.shape[0]

        Q = torch.zeros((N_pupil,N_pupil,2,2), dtype=torch.cfloat)
        Q[...,0,0] = self.qs[0] + 1j*self.qs[3]
        Q[...,0,1] = self.qs[2] + 1j*self.qs[1]
        Q[...,1,0] = -self.qs[2] + 1j*self.qs[1]
        Q[...,1,1] = self.qs[0] - 1j*self.qs[3]

        Gamma = aperture*torch.exp(1j*2*np.pi*self.W)

        return (Gamma[...,None,None] * Q)


class torchScalarPixels(torchScalarWindow):

    def __init__(self, aperture_size=.99, computation_size=4., N_pts=128):
        super(torchScalarPixels, self).__init__(aperture_size, computation_size, N_pts)
        
        aperture = self.get_aperture(dummy_ind=0)
        # n_opt_pix = len(aperture[aperture==1])

        self.phase = nn.Parameter(torch.zeros(aperture.shape, requires_grad=True, dtype=torch.float))
        self.A = nn.Parameter(torch.tensor(1., requires_grad=True, dtype=torch.float))
        
    
    def get_pupil_array(self):
        

        return self.get_aperture(dummy_ind=0)*self.A*torch.exp(1j*2*np.pi*self.phase)
    


class torchApodizedUnitary(torchBirefringentWindow):

    def __init__(self, c_A=None, c_W=None, c_q=None, aperture_size=.99, computation_size=4., 
                 N_pts=128, jmax_list=[15]*5 +[1], index_convention='standard'
                 ):
        super(torchApodizedUnitary, self).__init__(aperture_size, computation_size, N_pts)
        
        self.jmax=jmax_list
        if c_A is None:
            tempA = torch.zeros((jmax_list[5]), dtype=torch.float)
            tempA[0] = 1
            self.c_A = nn.Parameter(tempA, requires_grad=True)
        else:
            self.c_A = torch.nn.Parameter(torch.from_numpy(c_A).type(torch.float), requires_grad=True)
            self.jmax[-1]=len(c_A)

        cq_list = []
        if c_q is None:
            tempq = torch.zeros((jmax_list[0]), dtype=torch.float)
            tempq[0] = 1
            cq_list = [nn.Parameter(tempq, requires_grad=True)]
            for i in range(1,4):
                cq_list += [nn.Parameter(torch.zeros(jmax_list[i], requires_grad=True, dtype=torch.float))]
        else:
            cq_list=[]
            for i in range(0,4):
                cq_list += [torch.nn.Parameter(torch.from_numpy(c_q[i]).type(torch.float), requires_grad=True)]
                self.jmax[i] = len(c_q[i])
        self.c_q = nn.ParameterList(cq_list)
        
        if c_W is None:
            self.c_W = nn.Parameter(torch.zeros(jmax_list[4]-2, requires_grad=True, dtype=torch.float))
        else:
            self.c_W = torch.nn.Parameter(torch.from_numpy(c_W).type(torch.float), requires_grad=True)
            self.jmax[-2]=len(c_W)+2

        self.index_convention = index_convention                  
        self.defocus_j = defocus_j(index_convention)
    
    def get_pupil_array(self):
        ux, uy = self.xy_mesh()
        N_pupil = ux.shape[0]
        zernike_seq = zernike_sequence(np.max(self.jmax), 
                                        self.index_convention, 
                                        ux/self.aperture_size, 
                                        uy/self.aperture_size)

        qs = torch.zeros((4,N_pupil,N_pupil),  dtype=torch.cfloat)
        for q_ind in range(4):
            qs[q_ind] = torch.sum(self.c_q[q_ind]*zernike_seq[...,:self.jmax[q_ind]],-1)
        Q = torch.zeros((N_pupil,N_pupil,2,2), dtype=torch.cfloat)
        Q[...,0,0] = qs[0] + 1j*qs[3]
        Q[...,0,1] = qs[2] + 1j*qs[1]
        Q[...,1,0] = -qs[2] + 1j*qs[1]
        Q[...,1,1] = qs[0] - 1j*qs[3]
        normQ = torch.sqrt(torch.sum(torch.abs(qs)**2, 0))
        # normQ[normQ==0] = 1

        Amp = torch.sum(self.c_A*zernike_seq[...,:self.jmax[-1]],-1)
        W = torch.sum(zernike_seq[...,1:self.defocus_j]*self.c_W[:self.defocus_j-1],-1)\
            + torch.sum(zernike_seq[...,self.defocus_j+1:self.jmax[-2]]*
                        self.c_W[self.defocus_j-1:],-1)
        Gamma = self.get_aperture(dummy_ind=0)*Amp*torch.exp(1j*2*np.pi*W)

        return ( Gamma[...,None,None] * Q/ normQ[...,None,None])