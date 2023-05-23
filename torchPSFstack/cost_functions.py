"""Module containing the definition of the cost functions"""
import torch

def loss_loglikelihood(output, target):
    """Cost function for Poisson noise.
    
    Parameters
    ----------
    output : Tensor
        Modeled PSF stack.
    target : Tensor
        Measured or target PSF stack.

    Returns
    -------
    loss : Tensor
        Value for the cost function. 
    """
    loss = torch.mean(-target * torch.log(output) + output)
    # loss = torch.mean(-target * torch.log(output/target) + output-target)
    return loss

def loss_sumsquared(output, target):
    """Cost function for Gaussian noise.
    
    Parameters
    ----------
    output : Tensor
        Modeled PSF stack.
    target : Tensor
        Measured or target PSF stack.
        
    Returns
    -------
    loss : Tensor
        Value for the cost function. 
    """
    loss = torch.mean( (output-target)**2 )
    return loss