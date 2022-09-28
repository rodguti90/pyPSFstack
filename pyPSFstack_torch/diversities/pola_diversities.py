import torch


class torchPDiversity():
    def __init__(self, *args):
        self.jones_list = self.get_jones_list(*args)
        self.N_pdiv = len(self.jones_list)

    def get_jones_list(self):
        raise NotImplementedError("Please Implement this method")

    def forward(self, input):
        return self.jones_list @ input[...,None,:,:]

class torchPDiversity_QWP(torchPDiversity):

    def __init__(self, angles):
        torchPDiversity.__init__(self, angles)

    def get_jones_list(self, angles):
        return jones_qwp(angles)

class torchPDiversity_LP(torchPDiversity):

    def __init__(self, angles):
        torchPDiversity.__init__(self, angles)
        
    def get_jones_list(self, angles):
        return jones_lp(angles)

class torchPDiversity_Compound(torchPDiversity):
    def __init__(self, pdivs):
        torchPDiversity.__init__(self, pdivs)

    def get_jones_list(self, pdivs):
        n_pdivs = len(pdivs)
        jones = pdivs[0].jones_list
        for n_p in range(1,n_pdivs):
            sh = list(pdivs[n_p].jones_list.shape)
            jones = torch.reshape(pdivs[n_p].jones_list, [sh[0]]+[1]*n_p+sh[1:])\
                @ jones
        return jones.reshape(-1,2,2)

def jones_qwp(angles):
    theta = torch.tensor(angles)
    n_a = len(theta)
    jones =torch.empty((n_a,2,2), dtype=torch.cfloat)
    jones[...,0,0] = torch.cos(theta)**2 + 1j * torch.sin(theta)**2
    jones[...,0,1] = (1-1j)*torch.sin(theta)*torch.cos(theta)
    jones[...,1,1] = 1j*torch.conj(jones[...,0,0])
    jones[...,1,0] = jones[...,0,1]
    return torch.exp(torch.tensor(-1j*torch.pi/4)) * jones

def jones_lp(angles):
    theta = torch.tensor(angles)
    n_a = len(theta)
    jones =torch.empty((n_a,2,2), dtype=torch.cfloat)
    jones[...,0,0] = torch.cos(theta)**2
    jones[...,0,1] = torch.sin(theta)*torch.cos(theta)
    jones[...,1,1] = torch.sin(theta)**2
    jones[...,1,0] = jones[...,0,1]
    return jones

def jones_gwp(angles, eta):
    theta = torch.tensor(angles)
    eta = torch.tensor(eta)
    n_a = len(theta)
    jones =torch.empty((n_a,2,2), dtype=torch.cfloat)
    jones[...,0,0] = torch.exp(-1j*eta/2) * torch.cos(theta)**2 + \
        torch.exp(1j*eta/2) * torch.sin(theta)**2
    jones[...,0,1] = torch.exp(-1j*eta/2)*(1-torch.exp(1j*eta))*torch.sin(theta)*torch.cos(theta)
    jones[...,1,1] = torch.conj(jones[...,0,0])
    jones[...,1,0] = jones[...,0,1]
    return jones