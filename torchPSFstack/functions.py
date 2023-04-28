import torch
import numpy as np
from colorsys import hls_to_rgb
from skimage.morphology import erosion, dilation

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

def crop_center(input, size):
    x = input.shape[0]
    y = input.shape[1]
    start_x = x//2-(size//2)
    start_y = y//2-(size//2)
    return input[start_x:start_x+size,start_y:start_y+size,...]



def get_pupils_param_dict(model):
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
    
    
    if 'pb_bck' in md_list:
        dic['pb_bck'] = {}
        for name in model.pb_bck.state_dict().keys():
            dic['pb_bck'][name] =  (model.state_dict()['pb_bck'+'.'+name]).cpu().numpy()
            
    if 'blurring.bk' in md_list:
        dic['blurring'] = {}
        for name in model.blurring.bk.state_dict().keys():
            dic['blurring'][name] =  (model.state_dict()['blurring.bk'+'.'+name]).cpu().numpy()
            

    return dic


def outer_pixels(stack):
    shape_stack = list(stack.shape)
    [NX, NY] = np.meshgrid(np.arange(shape_stack[0]),
                           np.arange(shape_stack[1]))
    outer_pix = ((NX - (shape_stack[1]-1)/2)**2
                 + (NY - (shape_stack[0]-1)/2)**2
                 > ((np.min(shape_stack[:2])-1)/2)**2)

    return outer_pix.reshape(shape_stack[:2]+[1]*len(shape_stack[2:]))

def get_normNbck(stack):
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
    """
    ESTIMATE_BCKGD estimates the value of the background
    illumination and its standard deviation on the images by 
    computing the mean value on the pixels outside a circle of 
    radius NX/2         
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