#!/usr/bin/env python
"""
Laura Spitler
"""
import sys
import numpy as np
import numpy.random as random
import matplotlib.pyplot as py
from scipy.optimize import curve_fit

def TwoD_Gaussian((x,y), A, x0, y0, sx, sy, B):

    c=2*np.sqrt(np.log(2)*2)
    g = A*np.exp(-((x-x0)**2/(2*(sx/c)**2)+((y-y0)**2/(2*(sy/c)**2))))+B

    return g.ravel()
#    return g

def fake_2D_Gaussian(A, x0, y0, sx, sy, B, dim=(512,512), donoise=False):

    rr = np.arange(0, dim[0])
    cc = np.arange(0, dim[1])
    cc,rr = np.meshgrid(cc,rr)

    if donoise:
        noise=random.normal(0, 1.0, size=dim)
    else:
        noise=np.zeros(dim)

    gdata=TwoD_Gaussian((cc,rr), A, x0, y0, sx, sy, B)
    gdata.shape=dim
    data = noise + gdata

    return data

def fit_2D_Gaussian(data, guess=[10, 100, 100, 32, 32, 0], maxfev=1400):

    ydim,xdim=data.shape

    rr = np.arange(0, ydim)
    cc = np.arange(0, xdim)
    cc,rr = np.meshgrid(cc,rr)

    popt, pcov = curve_fit(TwoD_Gaussian, (cc,rr), data.ravel(), p0=guess, maxfev=maxfev)
    uncert=np.sqrt(np.diag(pcov))

    return popt, uncert

def plot_2D_Gaussian_fit(data, popt, ncontour=8):

    ydim,xdim=data.shape

    rr = np.arange(0, ydim)
    cc = np.arange(0, xdim)
    cc,rr = np.meshgrid(cc,rr)

    fitted_data=TwoD_Gaussian((cc,rr), *popt)

    py.figure()
    ax=py.subplot(111)
    ax.imshow(data, origin='lower', interpolation='nearest')
    ax.contour(cc,rr,fitted_data.reshape(ydim,xdim), ncontour, colors='w')
    py.show()

    return ax

def OneD_Gaussian(x, A, x0, sx, B):

    c=2*np.sqrt(np.log(2)*2)
    g = A*np.exp(-((x-x0)**2/(2*(sx/c)**2)))+B

    return g

def fit_1D_Gaussian(data, guess=[1,100,10,0], maxfev=1400, tres=1, xdum=None):

    if xdum == None:
        xdum=tres*np.arange(data.shape[0])

    try:
        popt, pcov = curve_fit(OneD_Gaussian, xdum, data, p0=guess, maxfev=maxfev)
        uncert=np.sqrt(np.diag(pcov))
    except RuntimeError:
        popt = np.zeros(4)
        uncert = np.zeros(4)

    return popt, uncert

def plot_1D_Gaussian_fit(data, popt, tres=1, xdum=None):

    if xdum == None:
        xdum=np.arange(data.shape[0])

    py.figure()
    py.plot(tres*xdum, data[xdum], 'b-')
    py.plot(tres*xdum, OneD_Gaussian(xdum, *popt), 'r-')
    py.show()

    return
