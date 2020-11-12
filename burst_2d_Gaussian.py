#####
# Written by Laura Spitler
#
# ** Change log **
# - 30 Oct 2020
# Consolidated model generating and fitting functions. 
# Added doc strings and comments.
# Added new function to report parameter values.
#####

import numpy as np
import matplotlib.pyplot as py

import psrchive_utils as pau

from astropy.modeling import models, fitting
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve

from matplotlib.ticker import MaxNLocator, ScalarFormatter

def gen_Gauss2D_model(peak_bins, peak_amps, f0=1600., bw=200., dt=2.0 ):
    '''
    Generates a 2D Gaussian astropy model for LSQ fitting.
    The number of Gaussians included in the model is infered from the length of peak_bins.
    The input parameters give the initial guesses for the LSQ fitting.

    Note: the units of the time and frequency data can be sec and MHz or time bins and channels

    Function parameters:
    peak_bins = list of peak bins or times to give as an initial guess
    peak_amps = list of peak amplitudes to give as an initial guess
    f0 = center frequeny for initial guess
    bw = bandwidth for initial guess
    dt = burst duration for initial guess

    Returns:
    astropy 2D Gaussian model object
    '''
    npeaks=len(peak_bins)

    #Loop over peaks and add Gaussians to model
    for ii in range(npeaks):
        if ii == 0: g_guess = models.Gaussian2D(peak_amps[ii], x_mean=peak_bins[ii], y_mean=f0, \
                                            x_stddev=dt, y_stddev=bw, theta=0.0)
        else: g_guess += models.Gaussian2D(peak_amps[ii], x_mean=peak_bins[ii], y_mean=f0, \
                                            x_stddev=dt, y_stddev=bw, theta=0.0)
    return g_guess

def fit_Gauss2D_model(data, tdum, fdum, g_guess):
    '''
    Fits the 2D data provided to the astropy 2D Gaussian model object
      using the LM least squares routine

    Function parameters:
    data = 2D data of burst (format is freq. x time)
    tdum = dummy array along time dimension for LSQ fitting (can be seconds or bins).
        It should have the same length as axis=1 of data.
    fdum = dummy array along the frequency dimension for LSQ fitting (can be Hz or chan. num)
        It should have the same length as axis=0 of data.
    
    '''
    xdum,ydum = np.meshgrid(tdum,fdum)

    fit_LM = fitting.LevMarLSQFitter()
    fit_2dG = fit_LM(g_guess, xdum, ydum, data)

    return fit_2dG, fit_LM

def report_Gauss_parameters(best_gauss, fitter, verbose=False):
    '''
    Simple function to extract and report best fit parameters of Gaussian model

    Input:
    best_gauss = 2D Gaussian astropy modeling object
    verbose = option to print paramters to screen (default False)

    Return
    bparams = array of Gaussian parameters (dimension: npeaks x 6 parameters)
    '''
    npks = len(best_gauss.parameters)/6
    npks = int(npks)
    bparams = best_gauss.parameters.reshape((npks,6))

    #Pull out uncertainties
    cov_mat=fitter.fit_info['param_cov']
    bunc=np.sqrt(np.diag(cov_mat))
    print(bunc)
    bunc.shape=((npks,6))

    if verbose:
        for ii in range(npks):

            print("**** Sub-burst %d ****" % ii)
            print("Amp  : %.2f +/- %.2f" % (bparams[ii, 0], bunc[ii,0]))
            print("T Loc: %.2f +/- %.2f" % (bparams[ii, 1], bunc[ii,1]))
            print("F Loc: %.2f +/- %.2f" % (bparams[ii, 2], bunc[ii,2]))
            print("T Wid: %.2f +/- %.2f" % (bparams[ii, 3], bunc[ii,3]))
            print("F Wid: %.2f +/- %.2f" % (bparams[ii, 4], bunc[ii,4]))
            print("Angle: %.2f +/- %.2f" % (bparams[ii, 5], bunc[ii,5]))
            print("")

    return bparams, bunc

def dynspec_3pan(xarr, yarr, data, vlim=(-1,-1), tslim=(-1,-1), bplim=(-1,-1), title=''):
    '''Method to produce the three panel
    dynamic spectrum plot. The main panel
    is a pcolormesh figure of data with 
    the mesh defined by xarr and yarr.
    The bottom panel showes a time series of
    the 2D data, and the right panel shows 
    the average bandpass
    Input:
    xarr = 1D array representing xaxis.
       Used to generate mesh
    yarr = same as xarr but for yaxis
    data = 2D array of data
    Output:
    Figure instance
    '''
    tsize=14

    #Calculate the time series and average bandpass
    #  for the subpanel plots
    tseries = np.mean(data, axis=0)
    bandpass = np.mean(data, axis=1)

    #Convert to SNR units
    #inds,dstd,dmean=plsr.threshold(tseries, 4.0)
    dmedian = np.median(tseries)
    dstd = np.std(tseries)
    tseries=(tseries-dmedian)/dstd

    #If no plot limits specified, 
    if vlim==(-1,-1):
        vlim=(np.min(data), np.max(data))

    #Create figure instance, add axes and turn off labels
    fig=py.figure(figsize=(12,9))

    ax1 = fig.add_axes([0.1, 0.1, 0.6, 0.6])
    ax2 = fig.add_axes([0.1, 0.6, 0.6, 0.2], sharex=ax1)
    ax3 = fig.add_axes([0.7, 0.1, 0.2, 0.5], sharey=ax1)

    for i in ax2.get_xticklabels(): i.set_visible(False)
    for i in ax3.get_yticklabels(): i.set_visible(False)

    #Generage 2D mesh
    T,F=np.meshgrid(xarr,yarr)

    #Plot data
    ax1.pcolormesh(T,F,data, vmin=vlim[0], vmax=vlim[1])
    ax2.plot(xarr, tseries, 'y-')
    ax3.step(bandpass, yarr, 'y-')

    #Add labels to axes and title
    ax1.set_xlabel('$\\rm Time\; (ms)$', size=tsize)
    ax1.set_ylabel('$\\rm Frequency\; (MHz)$', size=tsize)
    ax1.set_title(title)

    ax2.set_ylabel('$\\rm S/N$', size=tsize)
    ax3.set_title('$\\rm Average\; Bandpass$', size=tsize)

    #Do some formatting of axes
    ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax3.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, prune='lower'))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, prune='lower'))

    #Set axis ranges
    ax1.set_xlim((min(xarr), max(xarr)))
    ax1.set_ylim((min(yarr), max(yarr)))

    return fig

def plot_burst_windows(stimes, freqs, data, best_gauss, ncontour=8, res_plot=False):
    '''Generates the usual dynamic spectrum plot using dynspec_3pan
    with the contours from the best-fit 2D Gaussian model overplotted

    Input parameters:
    stimes = dummy array for the time dimension
    freqs = dummy array for the freq dimension
    data = 2D burst time-frequency data
    best_gauss = astropy fitting object containing the best-fit LSQ fitted parameters
    ncontour = number of contour lines to plot (default 8)
    res_plot = if True, a residual plot will also be made (default False)

    '''
    T,F=np.meshgrid(stimes, freqs)

    #Make standard 3-panel dynamic spectrum plot
    fig=dynspec_3pan(stimes, freqs, data)

    #Add contours
    ax=fig.get_axes()
    ax[0].contour(T,F,best_gauss(T,F), ncontour, colors='w')

    #If requested, make the residual dynamic spectrum plot
    if res_plot:
        res_fig=dynspec_3pan(stimes, freqs, data-best_gauss(T,F))

    return
