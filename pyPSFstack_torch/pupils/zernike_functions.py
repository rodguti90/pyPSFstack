from math import factorial
import torch
import numpy as np



def cart2pol(x,y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return rho, phi

def radial_zernike(n,m,rho):
    #check that n-m is even
    assert (n-m)%2 == 0 & isinstance(n,int) & isinstance(m,int), \
        'n and m need to be ints and n-m even'
    pol = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        pol += (((-1)**k) * factorial(n - k) /
                (factorial(k) * factorial((n + m) // 2 - k) *
                 factorial((n - m) // 2 - k)))*rho**(n-2*k)
    return pol

def zernike_Z(n,m,x,y):
    assert x.shape == y.shape
    rho, phi = cart2pol(x,y)
    if m > 0:
        polZ = radial_zernike(n,m,rho)*torch.cos(m*phi)
    elif m < 0:
        mm = -m
        polZ = radial_zernike(n,mm,rho)*torch.sin(mm*phi)
    else:
        polZ = radial_zernike(n,m,rho)
    polZ[rho>1] = 0
    return np.sqrt((2*n+2)/np.pi)*polZ

def zernike_sequence(j_max, convention, x, y):
    y_size, x_size = x.shape 
    z_seq = torch.empty((y_size,x_size,j_max))
    for j in range(j_max):
        if convention == 'fringe':
            n, m = fringe2nm(j)
        elif convention == 'standard':
            n, m = standard2nm(j)
        else:
            raise NotImplementedError("The chosen convention is not implemented")
        z_seq[...,j] = zernike_Z(n,m,x,y) 
    return z_seq

def defocus_j(convention):
    if convention == 'fringe':
        return 3
    elif convention == 'standard':
        return 4 
    else:
        raise NotImplementedError("The chosen convention is not implemented")
        
def standard2nm(j):
    n = np.ceil((-3+np.sqrt(9+8*j))/2)
    m = 2*j - n*(n+2)
    return int(n), int(m)

def fringe2nm(j):
    for n in range(j+1):
        for m in range(-n,n+2,2):
            if m < 0:
                s = 1
            else:
                s = 0
            j_test = (1+(n+np.abs(m))/2)**2 -2*np.abs(m)+s -1
            if j == j_test: return n, m