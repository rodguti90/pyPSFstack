"""Module defining the Zernike polynomials and auxilary functions."""
from math import factorial
import numpy as np

def cart2pol(x,y):
    """Transform cartesian array into polar.
    
    Parameters
    ----------
    x : ndarray
        Array for the x coordinate.
    y : ndarray
        Array for the y coordinate.

    Returns
    -------
    rho : ndarray
        Array for the radial coordinate.
    phi : ndarray
        Array for the azimuthal coordinate.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def radial_zernike(n,m,rho):
    """Computes the radial Zenike polynomial.

    Parmaters
    ---------
    n : int
        The radial index.
    m : int
        Azimuthal index
    rho : ndarray
        Radial variable

    Returns
    -------
    pol : ndarray
        Array containing the radial Zernike polynomial.
    """
    #check that n-m is even
    assert (n-m)%2 == 0 & isinstance(n,int) & isinstance(m,int), \
        'n and m need to be ints and n-m even'
    pol = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        pol += (((-1)**k) * factorial(n - k) /
                (factorial(k) * factorial((n + m) // 2 - k) *
                 factorial((n - m) // 2 - k)))*rho**(n-2*k)
    return pol

def zernike_Z(n,m,x,y):
    """Computes the Zenike polynomial.

    Parmaters
    ---------
    n : int
        The radial index.
    m : int
        Azimuthal index
    x : ndarray
        Array for the x coordinate.
    y : ndarray
        Array for the y coordinate.

    Returns
    -------
    polZ : ndarray
        Array containing the radial Zernike polynomial.
    """
    assert x.shape == y.shape
    rho, phi = cart2pol(x,y)
    if m > 0:
        polZ = radial_zernike(n,m,rho)*np.cos(m*phi)
    elif m < 0:
        mm = -m
        polZ = radial_zernike(n,mm,rho)*np.sin(mm*phi)
    else:
        polZ = radial_zernike(n,m,rho)
    polZ[rho>1] = 0
    return np.sqrt((2*n+2)/np.pi)*polZ

def zernike_sequence(j_max, convention, x, y):
    """Computes a all Zernike polynomials with index < j_max.

    Parameters
    ----------
    j_max : int
        Number of Zernikes to include in teh sequence.
    convention : {'standard','fringe'}
        Defines which single index convention to use for the Zernike polynomials.
    x : ndarray
        Array for the x coordinate.
    y : ndarray
        Array for the y coordinate. 

    Returns
    -------
    z_seq : ndarray
        Array of j_max Zernike polynomials evaluated at points x and y.
    """
    y_size, x_size = x.shape 
    z_seq = np.empty((y_size,x_size,j_max))
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
    """Returns the index for defocus in a single index given convention.
    
    Parameters
    ----------
    convention : {'standard','fringe'}

    Returns
    -------
    int 
        Index for defocus Zernike polynomial for the chosen convention.
    """
    if convention == 'fringe':
        return 3
    elif convention == 'standard':
        return 4 
    else:
        raise NotImplementedError("The chosen convention is not implemented")
        
def standard2nm(j):
    """Gives the two indices n and m corresponding to the index for standard convention.
    
    Parameters
    ----------
    j : int 
        Index for standard convention

    Returns
    -------
    n : int
        Radial index.
    m : int
        Azimuthal index.
    """
    n = np.ceil((-3+np.sqrt(9+8*j))/2)
    m = 2*j - n*(n+2)
    return int(n), int(m)

def fringe2nm(j):
    """Gives the two indices n and m corresponding to the index for fringe convention.
    
    Parameters
    ----------
    j : int 
        Index for fringe convention

    Returns
    -------
    n : int
        Radial index.
    m : int
        Azimuthal index.
    """
    for n in range(j+1):
        for m in range(-n,n+2,2):
            if m < 0:
                s = 1
            else:
                s = 0
            j_test = (1+(n+np.abs(m))/2)**2 -2*np.abs(m)+s -1
            if j == j_test: return n, m

# def noll2mn(j):
