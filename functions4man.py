from turtle import color, width
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
from tqdm import tqdm

from pyPSFstack.core import PSFStack
from pyPSFstack.functions import trim_stack

from torchPSFstack.psf_modules import torchPSFStack, torchPSFStackTilts, torchTilts,torchDefocuses
from torchPSFstack.pupils.sources import torchDipoleInterfaceSource
from torchPSFstack.pupils.windows import torchDefocus, torchSEO
from torchPSFstack.pupils.aberrations import torchScalarAberrations, \
    torchUnitaryAberrations, torchApodizedUnitary, torchScalarPixels
from torchPSFstack.diversities.pupil_diversities import torchZDiversity
from torchPSFstack.diversities.pola_diversities import torchPDiversity_QWP, \
    torchPDiversity_LP, torchPDiversity_Compound, torchPDiversity_GWP
from torchPSFstack.blurring.blurring import torchNoBlurring
from torchPSFstack.cost_functions import loss_loglikelihood, loss_sumsquared

from torchPSFstack.functions import get_pupils_param_dict, get_normNbck, colorize


#############################################################################
# Plotting
#############################################################################

fig_w = 8#16
psf_cmap = 'inferno'
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })

def plot_zpstack(stack, N_p=5, off_mid=0):
    v_max = np.max(stack)
    v_min = np.min(stack)

    N_z, _ = stack.shape[-2:]
    mid_ind = N_z//2 + off_mid
    
    fig, axs = plt.subplots(3,N_p,figsize=(fig_w,3*fig_w/N_p), gridspec_kw={'wspace':.05, 'hspace':0.05})
    for ind in range(N_p):
        im = axs[0,ind].imshow(stack[...,0,ind],vmin=v_min,vmax=v_max,cmap=psf_cmap)
        im = axs[1,ind].imshow(stack[...,mid_ind,ind],vmin=v_min,vmax=v_max,cmap=psf_cmap)
        im = axs[2,ind].imshow(stack[...,-1,ind],vmin=v_min,vmax=v_max,cmap=psf_cmap)
    for ax in axs.ravel():
        ax.set_axis_off()
    
    cb_ax = fig.add_axes([0.91,0.13,0.01,0.75])
    fig.colorbar(im, cax=cb_ax)

def plot_zstack(stack,N_z=5):
    v_max = np.max(stack)
    v_min = np.min(stack)

    n_z = stack.shape[-1]
    mid_ind = n_z//2
    in_ind = mid_ind - N_z//2
    fig, axs = plt.subplots(1,N_z,figsize=(fig_w,fig_w/N_z), gridspec_kw={'wspace':0.05, 'hspace':0})
    for ind in range(in_ind,in_ind+N_z):
        im = axs[ind-in_ind].imshow(stack[...,ind],vmin=v_min,vmax=v_max,cmap=psf_cmap)
    for ax in axs.ravel():
        ax.set_axis_off()  

    cb_ax = fig.add_axes([0.91,0.13,0.01,0.75])
    fig.colorbar(im, cax=cb_ax)

def plot_xyz(stack, orientation='horizontal'):
    v_max = np.max(np.reshape(stack,(4,-1)),axis=1).reshape(4,1,1,1)
    stack /= v_max
    n_s = stack.shape[1]
    sc=20#30
    
    arrowprops=dict(arrowstyle="<|-|>", color='white',linewidth=0.075*n_s)
    
    if orientation=='horizontal':
        fig, axs = plt.subplots(2,4, figsize=(2*fig_w/4,2*fig_w/8), gridspec_kw={'wspace':0.1, 'hspace':0}) 
        cb_ax = fig.add_axes([0.91,0.14,0.017,0.73])
    elif orientation=='vertical':
        fig, axs = plt.subplots(4,2, figsize=(2*fig_w/8,2*fig_w/4), gridspec_kw={'wspace':0.1, 'hspace':0}) 
        axs = axs.T
        cb_ax = fig.add_axes([0.94,0.13,0.05,0.745])
    else:
        raise ValueError('Invalid option for orientation')
    
    for ind in range(4):
        im = axs[0,ind].imshow(stack[ind][...,0],vmin=0,vmax=1,cmap=psf_cmap)
        axs[0,ind].set_axis_off()
        # axs[0,ind].annotate("", xy=(n_s/sc, 3*n_s/sc), xytext=(10*n_s/sc, 3*n_s/sc),
        #     arrowprops=arrowprops)
        axs[1,ind].imshow(stack[ind][...,1],vmin=0,vmax=1,cmap=psf_cmap)
        axs[1,ind].set_axis_off()
        # axs[1,ind].annotate("", xy=(3*n_s/sc,n_s/sc), xytext=(3*n_s/sc, 10*n_s/sc),
        #     arrowprops=arrowprops)
    
    # cb_ax = fig.add_axes([0.91,0.14,0.017,0.73])
    fig.colorbar(im, cax=cb_ax)

    im.figure.axes[1].tick_params(axis="x", labelsize=18)

# def plot_xyz_vert(stack):
#     v_max = np.max(np.reshape(stack,(4,-1)),axis=1).reshape(4,1,1,1)
#     stack /= v_max
#     n_s = stack.shape[1]
#     sc=20
#     fig, axs = plt.subplots(4,2, figsize=(2*fig_w/8,2*fig_w/4), gridspec_kw={'wspace':0.1, 'hspace':0}) 
#     arrowprops=dict(arrowstyle="<|-|>", color='white',linewidth=0.075*n_s)
#     axs = axs.T
#     for ind in range(4):
#         im = axs[0,ind].imshow(stack[ind][...,0],vmin=0,vmax=1,cmap=psf_cmap)
#         axs[0,ind].set_axis_off()
#         axs[0,ind].annotate("", xy=(n_s/sc, 3*n_s/sc), xytext=(10*n_s/sc, 3*n_s/sc),
#             arrowprops=arrowprops)
#         axs[1,ind].imshow(stack[ind][...,1],vmin=0,vmax=1,cmap=psf_cmap)
#         axs[1,ind].set_axis_off()
#         axs[1,ind].annotate("", xy=(3*n_s/sc,n_s/sc), xytext=(3*n_s/sc, 10*n_s/sc),
#             arrowprops=arrowprops)
    
#     cb_ax = fig.add_axes([0.91,0.14,0.017,0.73])
#     fig.colorbar(im, cax=cb_ax)

def cat_mat(mat):
    assert len(mat.shape)>=4
    output = np.concatenate((mat[...,0],mat[...,1]), axis=1)
    output = np.concatenate((output[...,0],output[...,1]), axis=0)
    return output

def plot_jones(jones):
    fig, ax = plt.subplots(1,1, figsize=(fig_w/4,fig_w/4), gridspec_kw={'wspace':-.05, 'hspace':0})
    n_p = jones.shape[0]
    ax.imshow(colorize(cat_mat(jones)))
    ax.set_axis_off()
    # labels = [r'J${}_{xx}$',r'J\textsubscript{yx}',
    #     r'J\textsubscript{xy}',r'J\textsubscript{yy}']
    # for i in range(4):
    #     ax.text(((i//2)%2)*n_p,(i%2)*n_p +n_p/5, labels[i], color='white', fontsize=12)

def get_xyzstack(pupil_sequence, pdiv, N_stack=20):
    psf = PSFStack(pupil_sequence, pdiversity=pdiv)
    stack = []
    for dir in [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]:
        psf.compute_psf_stack(orientation=dir)
        stack += [ trim_stack(psf.psf_stack,N_stack)]
    return stack

#############################################################################
# Analysis
#############################################################################

def cpx_corr(Y1,Y2):
    corr = np.sum((Y1)*(Y2).conj())
    norm_fac = np.sqrt(np.sum(np.abs(Y1)**2)*np.sum(np.abs(Y2)**2))  
    return corr/norm_fac

def seq_cpx_corr(Yseq,Yref=None,mask=None):
    if Yref is None:
        Yref= Yseq[0]
    if mask is not None:
        Yref *= mask.flatten()
        Yseq *= mask.flatten()
    corr = np.sum(Yseq * Yref.conj(), axis=-1)
    norm_ref = np.sum(np.abs(Yref)**2)
    norm_seq = np.sum(np.abs(Yseq)**2, axis=-1)
    return corr/np.sqrt(norm_ref*norm_seq)

#############################################################################
# Pupil retrieval
#############################################################################

def find_pupil(data_stack, params, lr=3e-2, n_epochs = 200, 
               loss_fn=loss_loglikelihood, pdiv='qwplp', blurring=torchNoBlurring(), 
               opt_def=True, opt_delta=False, abe='unitary', opt_a=False,
               tilts=False, defocuses=False, seo=False):

    tsrc = torchDipoleInterfaceSource(**params['pupil'],**params['source'],opt_delta=opt_delta)
    tpupil_sequence = [tsrc]

    if opt_def:
        tdef = torchDefocus(**params['pupil'],**params['defocus'])
        tpupil_sequence += [tdef]

    if seo:
        tseo = torchSEO(**params['pupil'],**params['seo'], opt_params=True)
        tpupil_sequence += [tseo]

    if abe is not None:
        if abe=='unitary':
            twdw = torchUnitaryAberrations(**params['pupil'], **params['aberrations'])
        elif abe=='apodized':
            twdw = torchApodizedUnitary(**params['pupil'], **params['aberrations'])
        elif abe=='scalar':
            twdw = torchScalarAberrations(**params['pupil'], **params['aberrations'])
        elif abe=='scalar_pix':
            twdw = torchScalarPixels(**params['pupil'])
        else:
            raise ValueError('Option not implmented')
        tpupil_sequence += [twdw]
    
    
    if pdiv is not None:
        tzdiv = torchZDiversity(**params['zdiversity'], **params['pupil'])
        if pdiv=='qwplp':
            tpdiv = torchPDiversity_Compound([
                torchPDiversity_QWP(params['pdiversity']['qwp_angles']), 
                torchPDiversity_LP(params['pdiversity']['lp_angles'])])
        elif pdiv=='gwplp':
            tpdiv = torchPDiversity_Compound([
                torchPDiversity_GWP(params['pdiversity']['gwp_angles'],
                                    params['pdiversity']['eta']), 
                torchPDiversity_LP(params['pdiversity']['lp_angles'])])
        else:
            raise ValueError('Invalid pdiv option')
        
        sh_divs = [len(params['zdiversity']['z_list']), len(tpdiv.jones_list)]

    elif pdiv is None:
        tzdiv = torchZDiversity(**params['zdiversity_np'], **params['pupil'])
        tpdiv=None
        sh_divs = [len(params['zdiversity_np']['z_list'])]
    
    defs=None
    if defocuses:
        defs = torchDefocuses(sh_divs, **params['pupil'])

    if tilts:
        model_retrieved = torchPSFStackTilts(
                        data_stack.shape[0],
                        tpupil_sequence,
                        zdiversity=tzdiv,
                        pdiversity=tpdiv,
                        tilts=torchTilts(sh_divs, **params['pupil']),
                        defocuses=defs,
                        blurring=blurring
                        )
    else:
        model_retrieved = torchPSFStack(
                        data_stack.shape[0],
                        tpupil_sequence,
                        zdiversity=tzdiv,
                        pdiversity=tpdiv,
                        blurring=blurring
                        )



    data_norm, data_bck = get_normNbck(data_stack)
    with torch.no_grad():
        model_retrieved.eval()
        first_est = model_retrieved()
    model_retrieved.set_scale_factor(data_norm/torch.sum(first_est))
    model_retrieved.set_pb_bck(data_bck, opt_b=True, opt_a=opt_a)
    
    # with torch.no_grad():
    #     model_retrieved.eval()
    #     first_est = model_retrieved()


    optimizer = torch.optim.Adam(
        model_retrieved.parameters(), 
        lr=lr)
    
    data = torch.from_numpy(data_stack).type(torch.float)
    loss_evol =[]
    for epoch in tqdm(range(n_epochs)):
        
        model_retrieved.train()
        yhat = model_retrieved()
        
        loss = loss_fn(yhat, data)
        loss_evol += [loss.item()]
        loss.backward()    
        optimizer.step()
        optimizer.zero_grad()

    return model_retrieved, loss_evol





# def find_exp_pupil(data_stack, params, lr=3e-2, n_epochs = 200, loss_fn=loss_loglikelihood, 
#         blurring=torchNoBlurring(), opt_def=True,opt_delta=False, seo=False, abe=False, opt_a=False,
#         tilts=True, defocuses=True):

#     tsrc = torchDipoleInterfaceSource(**params['pupil'],**params['source'],opt_delta=opt_delta)
#     tpupil_sequence = [tsrc]

#     if opt_def:
#         tdef = torchDefocus(**params['pupil'],**params['defocus'])
#         tpupil_sequence += [tdef]
#     if seo:
#         tseo = torchSEO(**params['pupil'],**params['seo'], opt_params=True)
#         tpupil_sequence += [tseo]
#     if abe:
#         twdw = torchApodizedUnitary(**params['pupil'], **params['aberrations'])
#         # twdw = torchUnitaryAberrations(**params['pupil'], **params['aberrations'])
#         tpupil_sequence += [twdw]

#     tzdiv = torchZDiversity(**params['zdiversity'], **params['pupil'])
    
#     tpdiv = torchPDiversity_Compound([torchPDiversity_GWP(params['pdiversity']['gwp_angles'],
#                                                           params['pdiversity']['eta']), 
#             torchPDiversity_LP(params['pdiversity']['lp_angles'])])
#     sh_divs = [len(params['zdiversity']['z_list']), len(tpdiv.jones_list)]
    
#     defs=None
#     if defocuses:
#         defs = torchDefocuses(sh_divs, **params['pupil'])

#     if tilts:
#         model_retrieved = torchPSFStackTilts(
#                         data_stack.shape[0],
#                         tpupil_sequence,
#                         zdiversity=tzdiv,
#                         pdiversity=tpdiv,
#                         tilts=torchTilts(sh_divs, **params['pupil']),
#                         defocuses=defs,
#                         blurring=blurring
#                         )
#     else:
#         model_retrieved = torchPSFStack(
#                         data_stack.shape[0],
#                         tpupil_sequence,
#                         zdiversity=tzdiv,
#                         pdiversity=tpdiv,
#                         blurring=blurring
#                         )

#     data_norm, data_bck = get_normNbck(data_stack)
#     with torch.no_grad():
#         model_retrieved.eval()
#         first_est = model_retrieved()
#     model_retrieved.set_scale_factor(data_norm/torch.sum(first_est))
#     model_retrieved.set_pb_bck(data_bck, opt_b=True, opt_a=opt_a)
#     with torch.no_grad():
#         model_retrieved.eval()
#         first_est = model_retrieved()


#     optimizer = torch.optim.Adam(
#         model_retrieved.parameters(), 
#         lr=lr
#     )
#     data = torch.from_numpy(data_stack).type(torch.float)
#     loss_evol =[]
#     for epoch in tqdm(range(n_epochs)):
        
#         model_retrieved.train()
#         yhat = model_retrieved()
        
#         loss = loss_fn(yhat, data)
#         loss_evol += [loss.item()]
#         loss.backward()    
#         optimizer.step()
#         optimizer.zero_grad()

#     return model_retrieved, loss_evol