#!/bin/env python

import numpy
from numpy import inf
import matplotlib.pyplot as py
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib import rc

#rc('text', usetex=True)

def threshold(ts, thr,  verbose=False):
        #Thresholds data with sigma clipping
        #  The time series is cleaned until the ratio of subsequent
        #  cleanings is less than 1%
        #ts = time series
        #thr = multiplier of std dev for thresholding
        #verbose = report mean, std deviation, ratio for each inter

        nloops=0
        #Make local copy of ts to clean
        tstmp=numpy.copy(ts)

        #Calculate statistics of uncleaned data
        sigold=numpy.std(tstmp)
        mold=numpy.mean(tstmp)


        #Do first clean stage
        inds=numpy.where(tstmp-mold > thr*sigold)[0]
        tstmp[inds]=mold
        sig=numpy.std(tstmp)
        m=numpy.mean(tstmp)

        if verbose == True: print mold, sigold
        if verbose == True: print len(inds)
        if verbose == True: print m, sig, sigold/sig-1.0

        #Loop through cleaning stages until ratio of
        #  current stddev is less than 1 % the previous
        while sigold/sig - 1.0 > 0.01:

                inds=numpy.where(tstmp-m > thr*sig)[0]
                tstmp[inds]=m

                sigold=sig
                sig=numpy.std(tstmp)
                mold=m
                m=numpy.mean(tstmp)
                nloops+=1
                if verbose==True: print m, sig, sigold/sig-1.0

        #Now threshold with 'clean' statistics
        m,sig=mold,sigold
        inds=numpy.where(ts-m > thr*sig)[0]

        return inds,sig,m

def dynspec_3pan(xarr, yarr, data, cpfit='', annotate=False, vlim=(-1,-1), tslim=(-1,-1), bplim=(-1,-1), title=''):
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
    tseries=numpy.mean(data, axis=0)
    bandpass=numpy.mean(data, axis=1)

    #Convert to SNR units
    inds,dstd,dmean=threshold(tseries, 4.0)
    tseries=(tseries-dmean)/dstd

    #If no plot limits specified,
    if vlim==(-1,-1):
        vlim=(numpy.min(data), numpy.max(data))
    if tslim==(-1,-1):
        tslim=(numpy.min(tseries), numpy.max(tseries))
    if bplim==(-1,-1):
        bplim=(numpy.min(bandpass), numpy.max(bandpass))

    #Create figure instance, add axes and turn off labels
    fig=py.figure(figsize=(12,9))
    ax1 = fig.add_axes([0.1, 0.3, 0.6, 0.6])
    ax2 = fig.add_axes([0.1, 0.1, 0.6, 0.2], sharex=ax1)
    ax3 = fig.add_axes([0.7, 0.3, 0.2, 0.6], sharey=ax1)
    if annotate:
        ax4 = fig.add_axes([0.72, 0.08, 0.2, 0.17])
        ax4.annotate('Location of center:\n%.1f +/- %.2f ms\n%.1f +/- %.1f MHz'\
         % (cpfit[1]*1e3, cpfit[2]*1e3, cpfit[3], cpfit[4]), xy=(0,0.4))
        ax4.annotate('Widths of fit:\n%.2f +/- %.2f ms\n%.1f +/- %.1f MHz'\
        % (cpfit[5]*1e3, cpfit[6]*1e3, cpfit[7], cpfit[8]), xy=(0,0))
        ax4.axis('off')

    for i in ax3.get_yticklabels(): i.set_visible(False)
    for i in ax3.get_xticklabels(): i.set_rotation(270)
    for i in ax1.get_xticklabels(): i.set_visible(False)

    #Generage 2D mesh
    T,F=numpy.meshgrid(xarr,yarr)

    #Add plots
    ax1.pcolormesh(T,F,data, vmin=vlim[0], vmax=vlim[1])
    if len(xarr) > 100:
        ax2.plot(xarr, tseries, 'r.')
        ax2.plot(xarr, tseries, 'k-')
    else:
        ax2.plot(xarr, tseries, 'r-', drawstyle='steps')

    ax3.step(bandpass, yarr, 'g-')

    #Try and guess at the axes labels
    if xarr.dtype=='int32' or xarr.dtype=='int64':
        ax2.set_xlabel('$\\rm Sample\; number$', size=tsize)
    else:
        ax2.set_xlabel('$\\rm Time\; (sec)$', size=tsize)

    if yarr.dtype=='int32' or yarr.dtype=='int64':
        ax1.set_ylabel('$\\rm Frequency \; channel$', size=tsize)
    else:
        ax1.set_ylabel('$\\rm Frequency\; (MHz)$', size=tsize)

    ax2.set_ylabel('$\\rm S/N$', size=tsize)
    ax3.set_xlabel('$\\rm Average\; Bandpass$', size=tsize)
    ax1.set_title(title)

    #Do some formatting of axes
    ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax3.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
    ax3.xaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True, prune='lower'))

    ax1.set_xlim((min(xarr), max(xarr)))
    ax1.set_ylim((min(yarr), max(yarr)))
    ax2.set_ylim((tslim[0], tslim[1]))
    ax3.set_xlim(bplim)

    return fig
