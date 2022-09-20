import numpy as np

# from scipy.optimize import minimize
# from skimage.morphology import erosion, dilation


def dag(array):
    return np.conj(np.swapaxes(array,-2,-1))
    
def trim_stack(stack, N_new):
    shape_stack = stack.shape
    N_pts = shape_stack[0]
    trimmed_stack = stack[(N_pts-N_new)//2:(N_pts+N_new)//2,
                          (N_pts-N_new)//2:(N_pts+N_new)//2]
    return trimmed_stack



# def outer_pixels(stack):
#     shape_stack = list(stack.shape)
#     [NX, NY] = np.meshgrid(np.arange(shape_stack[0]),
#                            np.arange(shape_stack[1]))
#     outer_pix = ((NX - (shape_stack[1]-1)/2)**2
#                  + (NY - (shape_stack[0]-1)/2)**2
#                  > ((np.min(shape_stack[:2])-1)/2)**2)

#     return outer_pix.reshape(shape_stack[:2]+[1]*len(shape_stack[2:]))

# def denoise_stack(stack):
#     outer_pix = outer_pixels(stack)
#     bckgd = estimate_background(stack, mask=outer_pix)
#     std = np.std(stack, axis=(0,1), where=outer_pix)
#     denoised_stack = stack - bckgd
#     threshold_mask = denoised_stack > std
#     threshold_mask = erosion(threshold_mask)
#     threshold_mask = dilation(threshold_mask)
#     denoised_stack = denoised_stack * threshold_mask
#     denoised_stack[denoised_stack<0] = 0
#     return denoised_stack, bckgd

# def estimate_background(stack,mask=None):
#     """
#     ESTIMATE_BCKGD estimates the value of the background
#     illumination and its standard deviation on the images by 
#     computing the mean value on the pixels outside a circle of 
#     radius NX/2         
#     """  
#     background = np.mean(stack, axis=(0,1), where=mask)
#     return background
   
# def estimate_photobleach_background(data, model=None):
#     data_stack, bckgd = denoise_stack(data)
#     amp_data = np.sum(data_stack, axis=(0,1))
#     amp_data = amp_data / amp_data[0,0]
#     if model is not None:
#         if model.shape[0] != data.shape[0]:
#             model = trim_stack(model, data.shape[0])
#         model_stack, _ = denoise_stack(model)
#         amp_model = np.sum(model_stack, axis=(0,1))
#         amp_model = amp_model / amp_model[0,0]
#     else:
#         model_stack = 1
#         amp_model = 1
#     photobleach_amplitudes = amp_data / amp_model
#     photobleach_amplitudes[photobleach_amplitudes>1]=1
#     scale = np.sum(photobleach_amplitudes*model_stack) \
#         / np.sum(data_stack)
#     return photobleach_amplitudes, scale, scale*bckgd



# def zeropad_stack(stack, N_new):
#     N_old = stack.shape[0]
#     pad_width = [(N_new-N_old)//2, int(np.ceil((N_new-N_old)/2))]
#     padded_stack = np.pad(stack, [pad_width]*2+[[0,0]]*(stack.ndim-2))
#     return padded_stack

