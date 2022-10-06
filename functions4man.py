import numpy as np
import matplotlib.pyplot as plt
import torch as torch
from tqdm import tqdm

from pyPSFstack.core import PSFStack
from pyPSFstack.functions import trim_stack

from pyPSFstack_torch.psf_modules import torchPSFStack
from pyPSFstack_torch.pupils.sources import torchDipoleInterfaceSource
from pyPSFstack_torch.pupils.windows import torchSEO
from pyPSFstack_torch.pupils.aberrations import torchUnitaryAberrations
from pyPSFstack_torch.diversities.pupil_diversities import torchZDiversity
from pyPSFstack_torch.diversities.pola_diversities import torchPDiversity_QWP, \
    torchPDiversity_LP, torchPDiversity_Compound
from pyPSFstack_torch.cost_functions import loss_loglikelihood, loss_sumsquared

from pyPSFstack_torch.functions import get_pupils_param_dict, get_normNbck, colorize

fig_w = 16
psf_cmap = 'inferno'
plt.rcParams['text.usetex'] = True


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

def plot_xyz(stack):
    v_max = np.max(np.reshape(stack,(4,-1)),axis=1).reshape(4,1,1,1)
    stack /= v_max
    fig, axs = plt.subplots(2,4, figsize=(2*fig_w/4,2*fig_w/8), gridspec_kw={'wspace':0.1, 'hspace':0})
    
    for ind in range(4):
        im = axs[0,ind].imshow(stack[ind][...,0],vmin=0,vmax=1,cmap=psf_cmap)
        axs[0,ind].set_axis_off()
        axs[1,ind].imshow(stack[ind][...,1],vmin=0,vmax=1,cmap=psf_cmap)
        axs[1,ind].set_axis_off()
    
    cb_ax = fig.add_axes([0.91,0.14,0.017,0.73])
    fig.colorbar(im, cax=cb_ax)

def plot_jones(jones):
    fig, axs = plt.subplots(2, 2, figsize=(fig_w/4,fig_w/4), gridspec_kw={'wspace':-.05, 'hspace':0})
    axs = axs.ravel()
    labels = [r'$\textbf{J}_\textbf{{xx}}$',r'$\textbf{J}_\textbf{{xy}}$',
        r'$\textbf{J}_\textbf{{yx}}$',r'$\textbf{J}_\textbf{{yy}}$']
    for i in range(4):
        axs[i].imshow(colorize(jones[...,i%2,(i//2)%2]))
        axs[i].set_axis_off()
        axs[i].text(0.01*fig_w/8,2*fig_w/8, labels[i], color='white', fontsize=13)

def get_xyzstack(pupil_sequence, pdiv):
    psf = PSFStack(pupil_sequence, pdiversity=pdiv)
    stack = []
    for dir in [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]:
        psf.compute_psf_stack(orientation=dir)
        stack += [ trim_stack(psf.psf_stack,30)]
    return stack

def find_pupil(data_stack, params, lr=3e-2, n_epochs = 200, loss_fn=loss_loglikelihood, pdiv=True):

    tsrc = torchDipoleInterfaceSource(**params['pupil'],**params['source'])
    twdw = torchUnitaryAberrations(**params['pupil'], **params['aberrations'])
    tpupil_sequence = [tsrc, twdw]
    tzdiv = torchZDiversity(**params['zdiversity'], **params['pupil'])
    if pdiv:
        tpdiv = torchPDiversity_Compound([torchPDiversity_QWP(params['pdiversity']['qwp_angles']), 
            torchPDiversity_LP(params['pdiversity']['lp_angles'])])
    else:
        tpdiv=None

    model_retrieved = torchPSFStack(
                    data_stack.shape[0],
                    tpupil_sequence,
                    zdiversity=tzdiv,
                    pdiversity=tpdiv
                    )

    data_norm, data_bck = get_normNbck(data_stack)
    with torch.no_grad():
        model_retrieved.eval()
        first_est = model_retrieved()
    model_retrieved.set_scale_factor(data_norm/torch.sum(first_est))
    model_retrieved.set_pb_bck(data_bck, opt_b=True)
    with torch.no_grad():
        model_retrieved.eval()
        first_est = model_retrieved()


    optimizer = torch.optim.Adam(
        model_retrieved.parameters(), 
        lr=lr
    )
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