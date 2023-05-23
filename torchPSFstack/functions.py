import torch
import numpy as np
from colorsys import hls_to_rgb
from skimage.morphology import erosion, dilation
import copy 


def crop_center(input, size):
    """Trims an array to anew one of reduced size.

    It assumes the image indices are the two frist ones.

    Parameters
    ----------
    input : ndarray, Tensor
        Stack of PSFs to be trimmed.
    size : int
        New size of the trimmed stack.

    Returns
    -------
    ndarray, Tensor
        Trimmed array
    """
    x = input.shape[0]
    y = input.shape[1]
    start_x = (x-size)//2
    start_y = (y-size)//2
    return input[start_x:start_x+size,start_y:start_y+size,...]



def get_pupils_param_dict(model):
    """Extracts the optimization parameters from the PSF model.
    
    Parameters
    ----------
    model : torchPSFStack module
        Module use to retrieve an unknown pupil from a PSF stack.

    Returns
    -------
    dic : dict
        Dictionnary containing all the optimization parameters organized as
        subdictionaries for each submodule of the PSFstack module.
    """
    dic = {}
    md_list = []
    for name in model.state_dict().keys():
        pt_ind = name.rfind('.')
        md_list += [name[:pt_ind]]


    for pupil_ind in range(len(model.pupils)):
        dic['pupil'+str(pupil_ind)] = {}
        state = model.pupils[pupil_ind].state_dict()
        
        curr_dic = dic['pupil'+str(pupil_ind)]
        for name in state.keys():
            pt_ind = name.rfind('.')
            if pt_ind>=0:
                if name[:pt_ind] in curr_dic.keys():
                    curr_dic[name[:pt_ind]] += [torch.Tensor.cpu(state[name]).numpy()]
                else:
                    curr_dic[name[:pt_ind]] = [torch.Tensor.cpu(state[name]).numpy()]
            else:
                curr_dic[name] = torch.Tensor.cpu(state[name]).numpy()
    
    if 'tilts' in md_list:
        dic['tilts'] = {}
        for name in model.tilts.state_dict().keys():
            dic['tilts'][name] =  (model.state_dict()['tilts'+'.'+name]).cpu().numpy()
    
    if 'pb_bck' in md_list:
        dic['pb_bck'] = {}
        for name in model.pb_bck.state_dict().keys():
            dic['pb_bck'][name] =  (model.state_dict()['pb_bck'+'.'+name]).cpu().numpy()
            
    if 'blurring.bk' in md_list:
        dic['blurring'] = {}
        for name in model.blurring.bk.state_dict().keys():
            dic['blurring'][name] =  (model.state_dict()['blurring.bk'+'.'+name]).cpu().numpy()
            

    return copy.deepcopy(dic)


def outer_pixels(stack):
    """Returns a boolean array selecting the corners of the PSF images in the stack.
    
    Parameters
    ----------
    stack : ndarray
        Stack of PSFs.
    
    Returns
    -------
    ndarray of bool
        Array being true for the four corners of the PSF images lying outside
        the disk of diameter equal to the size of the images.
    """
    shape_stack = list(stack.shape)
    [NX, NY] = np.meshgrid(np.arange(shape_stack[0]),
                           np.arange(shape_stack[1]))
    outer_pix = ((NX - (shape_stack[1]-1)/2)**2
                 + (NY - (shape_stack[0]-1)/2)**2
                 > ((np.min(shape_stack[:2])-1)/2)**2)

    return outer_pix.reshape(shape_stack[:2]+[1]*len(shape_stack[2:]))

def get_normNbck(stack):
    """Returns an estimation for the background illumination and the amplitude factor.

    Uses the four corners of the PSF images lying outside the disk of 
    diameter equal to the size of the images to estimate the background, 
    then subtracts it to estimate the overall amplitude factor.
    Parameters
    ----------
    stack : ndarray
        Stack of PSFs.
    
    Returns
    -------
    float
        Overall amplitude factor
    ndarray        
        Background estimation for each diversity.
    """
    outer_pix = outer_pixels(stack)
    bckgd = np.mean(estimate_background(stack, mask=outer_pix))
    std = np.std(stack, axis=(0,1), where=outer_pix)
    denoised_stack = stack - bckgd
    # threshold_mask = denoised_stack > std
    # threshold_mask = erosion(threshold_mask)
    # threshold_mask = dilation(threshold_mask)
    # denoised_stack = denoised_stack * threshold_mask
    denoised_stack[denoised_stack<0] = 0
    return np.sum(denoised_stack), bckgd

def estimate_background(stack,mask=None):
    """Estimates the value of the background using a specified region.

    Parameters
    ----------
    stack : ndarray
        Stack of PSFs.
    mask : ndarray of bool
        Array specifying the region used to estimate the background.
    
    Returns
    -------
    ndarray        
        Background estimation for each diversity.
    """  
    background = np.mean(stack, axis=(0,1), where=mask)
    return background

def colorize(z, 
             theme='dark', 
             saturation=1., 
             beta=1.4, 
             transparent=False, 
             alpha=1., 
             max_threshold=1):
    """Transforms a complex valued array into an rgb one.
    
    The phase is encoded as hue and the amplitude as lightness. 

    Parameters
    ----------
        z : ndarray
            Complex valued array.
        theme : {'dark', 'white'}, optional
            For 'dark' the lightness value tends to zero as the amplitude 
            diminishes and for 'white' it tends to one. 
        saturation : float
            Defines the saturation for hls
        beta : float
            Controls the scaling of lightness with respect to amplitude.
        transparent : bool, optional
            Whether to modify the alpha channel according to the amplitud.
        alpha : float
            Scaling for alpha channel controlling the opacity, 
            'transparent' must be set to True.
        max_threshold : float
            Can be used to change the range for shown for the amplitude.

    Returns
    -------
    c : ndarray
        Returned array transformed into rgb format. 
    """
    r = np.abs(z)
    r /= max_threshold*np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = np.transpose(c, (1,2,0))  
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c
    

# def cart2pol(x,y):
#     rho = torch.sqrt(x**2 + y**2)
#     phi = torch.atan2(y, x)
#     return rho, phi

# def xy_mesh(size, step):
#     u_vec = torch.arange(-size/2,
#                     size/2,
#                     step, dtype = torch.float32)
#     uy, ux = torch.meshgrid(u_vec,u_vec)
#     return ux, uy

# def polar_mesh(size, step):
#     ux, uy = xy_mesh(size, step)
#     ur = torch.sqrt(ux**2 + uy**2)
#     uphi = torch.atan2(uy, ux)
#     return ur, uphi
