"""Module containing the definition for all torchPupil abstract classes.
"""
import torch
import torch.nn as nn


class torchPupil(nn.Module):
    """torchPupil superclass.

    Attributes
    ----------
    aperture_size : float
        Normalized value for the aperture at the BFP i.e. NA/nf
    computation_size : float
        The total size at the BFP used for computation.
    N_pts : int
        Number of points used for the computation.
    step_f : float
        Step size of at the BFP.
    
    Methods
    -------
    get_pupil_array()
        Computes the pupil array.
    """
    def __init__(self, aperture_size, computation_size, N_pts):
        """Constructor.

        Parameters
        ----------
        aperture_size : float
            Normalized value for the aperture at the BFP i.e. NA/nf
        computation_size : float
            The total size at the BFP used for computation.
        N_pts : int
            Number of points used for the computation.
        """
        super(torchPupil, self).__init__()
        self.aperture_size = aperture_size 
        self.computation_size = computation_size
        self.N_pts = N_pts
        self.step_f = self.computation_size/self.N_pts

    def xy_mesh(self, umax=1):
        u_vec = torch.arange(-umax,
                        umax,
                        self.step_f, dtype = torch.float)
        ux, uy = torch.meshgrid(u_vec,u_vec, indexing='xy')
        return ux, -uy

    def polar_mesh(self, umax=1):
        ux, uy = self.xy_mesh(umax=umax)
        ur = torch.sqrt(ux**2 + uy**2)
        uphi = torch.atan2(uy, ux)
        return ur, uphi
    
    def get_aperture(self, umax=1, dummy_ind=2):
        ur, _ = self.polar_mesh(umax=umax)
        n_pts = ur.shape[0]
        aperture = torch.empty([n_pts,n_pts]+[1]*dummy_ind, dtype=torch.float)
        aperture = torch.reshape(ur**2, [n_pts,n_pts]+[1]*dummy_ind) \
            <= self.aperture_size**2
        return aperture

    def get_pupil_array(self):       
        """Computes the pupil array.
        
        Returns
        -------
        Tensor
            Complex torch tensor for the pupil sampled accordimng to its parameters.
        """  
        raise NotImplementedError("Please Implement this method")

class torchPupilDiversity(torchPupil):
    """torchPupil subclass for defining scalar diversities."""
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchPupilDiversity, self).__init__(aperture_size, computation_size, N_pts)

    def forward(self, input):
        pupil_array = self.get_pupil_array()
        dims = len(input.shape)
        sh = list(pupil_array.shape)
        output = input[..., None, :, :] \
            * torch.reshape(pupil_array, sh[:2]+[1]*(dims-4)+[sh[2]]+[1,1])
        return output

class torchSource(torchPupil):
    """torchPupil subclass for defining sources."""
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchSource, self).__init__(aperture_size, computation_size, N_pts)

    def forward(self):
        return self.get_pupil_array()

class torchScalarWindow(torchPupil):
    """torchPupil subclass for defining scalar windows."""
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchScalarWindow, self).__init__(aperture_size, computation_size, N_pts)

    def forward(self, input):
        pupil_array = self.get_pupil_array()
        return pupil_array[...,None,None] * input


class torchBirefringentWindow(torchPupil):
    """torchPupil subclass for birefringent windows."""
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchBirefringentWindow, self).__init__(aperture_size, computation_size, N_pts)

    def forward(self, input):
        return self.get_pupil_array() @ input

class torchBlurringKernel(torchPupil):
    """torchPupil subclass for defining blurring kernels."""
    def __init__(self, 
                 aperture_size=1., 
                 computation_size=4., 
                 N_pts=128):
        super(torchBlurringKernel, self).__init__(aperture_size, computation_size, N_pts)

    
    def forward(self, input):
        bk = self.get_pupil_array()
        dim_bk = len(bk.shape)
        dim_in = len(input.shape)
        # assert input.shape[:dim_bk] == bk.shape
        N_data = input.shape[0]

        bk = torch.reshape(bk, list(bk.shape)+[1]*(dim_in-dim_bk))
        otf = torch.fft.fftshift(torch.fft.ifft2(input, dim=(0,1), s=(self.N_pts,self.N_pts)), dim=(0,1))
        output = torch.fft.fft2(torch.fft.ifftshift(otf * bk, dim=(0,1)), 
                dim=(0,1))
        output = output[:N_data,:N_data,...]
        if dim_bk==3:
            output = torch.sum(output, axis=2)

        return output