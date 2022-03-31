import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

from sources import DipoleInterfaceSource
from windows import SEO
from aberrations import UnitaryPolarizationAberrations
from diversities import ZDiversity, PDiversity


class IncoherentPSF(nn.Module):

    def __init__(self,
                 aperture_size=.99, 
                 computation_size=4., 
                 N_pts=128, 
                 ni=1.33, 
                 nf=1.518, 
                 delta=0.1,
                 c=1.24*np.pi,
                 jmax=[15]*5, 
                 index_convention='fringe'
                 ):
        super(IncoherentPSF, self).__init__()

        self.source = DipoleInterfaceSource(
            aperture_size=aperture_size,
            computation_size=computation_size, 
            N_pts=N_pts, 
            ni=ni, 
            nf=nf, 
            delta=delta)

        self.window = SEO(aperture_size=aperture_size, 
                          computation_size=computation_size, 
                          N_pts=N_pts, 
                          c=c)

        self.aberrations = UnitaryPolarizationAberrations(
            aperture_size=aperture_size, 
            computation_size=computation_size, 
            N_pts=N_pts, 
            jmax=jmax, 
            index_convention=index_convention)

        self.zdiversity = ZDiversity(
            aperture_size=aperture_size, 
            computation_size=computation_size, 
            N_pts=N_pts, 
            nf=nf)

        self.pdiversity = PDiversity(pol_analyzer, angle_list=angle_list)

    def forward(self, z_list):

        input = self.source()
        input = self.window(input)
        input = self.aberrations(input)
        input = self.zdiversity(input, z_list)
        input = fft.fftshift(fft.fft2(
            fft.ifftshift(input,dim=(0,1)),dim=(0,1)),dim=(0,1))
        input = self.pdiversity(input)
        output = torch.sum(torch.abs(input)**2,dim=(-2,-1))