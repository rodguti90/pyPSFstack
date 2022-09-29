import torch
import numpy as np
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