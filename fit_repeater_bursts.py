#!/usr/bin/env python
"""
Laura Spitler
"""
import sys
import os.path
import numpy as np
import matplotlib.pyplot as py
import plotfuncs as pf
sys.path.append('/home/nimmo/AO_burst_properties_pipeline/')
import fit_2D_gaussian as f2Dg

def plot_burst_windows(stimes, freqs, data, cpfit, annotate=False, xedges=[], pfit=[], units='bins', ncontour=8):

    fig=pf.dynspec_3pan(stimes, freqs, data, cpfit, annotate)

    ax=fig.get_axes()

    if len(xedges) != 0:
        for ll,rr in xedges:
            ax[0].axvline(stimes[int(ll)], color='r')
            ax[0].axvline(stimes[int(rr)], color='b')
            ax[1].axvline(stimes[int(ll)], color='r')
            ax[1].axvline(stimes[int(rr)], color='b')

    if len(pfit) != 0:
        if units=='bins':
            ydim,xdim=data.shape
            rr=np.arange(0,ydim)
            cc=np.arange(0,xdim)
            cc,rr=np.meshgrid(cc,rr)
        else:
            cc,rr=np.meshgrid(stimes,freqs)

        fitted_data=f2Dg.TwoD_Gaussian((cc,rr), *pfit)
        T,F=np.meshgrid(stimes, freqs)

        ax[0].contour(T,F,fitted_data.reshape(len(freqs),len(stimes)), ncontour, colors='w')
        ax[0].contour(T,F,fitted_data.reshape(len(freqs),len(stimes)), 1, levels=[0.5*pfit[0]], colors='k')

    return fig

def plot_burst_residuals(stimes, freqs, data, pfit=[]):

    if len(pfit) != 0:
        ydim,xdim=data.shape
        rr=np.arange(0,ydim)
        cc=np.arange(0,xdim)
        cc,rr=np.meshgrid(cc,rr)


        fitted_data=f2Dg.TwoD_Gaussian((cc,rr), *pfit)
        T,F=np.meshgrid(stimes, freqs)

    fig=pf.dynspec_3pan(stimes, freqs, data-fitted_data.reshape(len(freqs), len(stimes)))

    return fig


def fit_burst_component(data, xedge=[0,0], yedge=[0,0], sig_frac=0.25, guess=[], maxfev=1400):

    #If no edge is given, assume full size of "data"
    if xedge == [0,0]:
        xedge = [0,data.shape[1]]
    if yedge == [0,0]:
        yedge = [0,data.shape[0]]

    subdata=np.copy(data[yedge[0]:yedge[1], xedge[0]:xedge[1]])

    #If no starting parameter for fit give, get some rough values
    if len(guess) == 0:

        #Calc spectrum of full array...
        spec=data.mean(axis=1)
        #... assume peak is location of component center
        ymax=spec.argmax()

        #Calc time series of full array...
        ts=data.mean(axis=0)
        #...assume peak is location of component center
        xmax=ts.argmax()

        #Max of all data roughly amplitude
        dmax=data[:,xmax].max()
        print "A guess; ", dmax
        print "x loc guess: ", xmax
        print "y loc guess: ", ymax

        #Assume burst x- and y-sigmas roughly "wid_frac" of width of data
        xwid=sig_frac*data.shape[1]
        ywid=sig_frac*data.shape[0]

        guess=[dmax, xmax+xedge[0], ymax+yedge[0], xwid, ywid, 0]

        print "x width guess: ", xwid
        print "y width guess: ", ywid

    else:
        print "Manual guess: ", guess

    #Do actual fit
    #Note, this returns fitted parameters in *bins*, not in physical units
    popt,uncert=f2Dg.fit_2D_Gaussian(data, guess=guess, maxfev=maxfev)

    return popt,uncert

def convert_fit(pfit, perr, stimes, freqs, tofile=False):

    #Calculate time and freq resolution from time and freq dummy arrays
    dt=stimes[1]-stimes[0]
    df=freqs[1]-freqs[0]

    #If an uncertity
    if np.isinf(perr[0]):
        for ii in range(len(perr)): perr[ii]=0

    ####EDIT: Values reported in terms of bins. Needed for my plotting script####
    tloc=stimes[0] + pfit[1]#*dt
    tloc_e=perr[1]#*dt
    floc=pfit[2]#*df
    floc_e=perr[2]#*df

    bin_center = [floc,tloc]

    twid=pfit[3]#*dt
    twid_e=perr[3]#*dt
    fwid=pfit[4]#*np.abs(df)
    fwid_e=perr[4]#*np.abs(df)
    print "Center bins: %.1f +/- %.2f , %.1f +/- %.1f fbins" % (tloc, tloc_e, floc, floc_e)
    print "Widths of fit in bins: %.2f +/- %.2f , %.1f +/- %.1f" % (twid, twid_e, fwid, fwid_e)
    bin_time = np.array([tloc,twid])

    #Convert location of center of Gaussian into physical units
    tloc=stimes[0] + pfit[1]*dt
    tloc_e=perr[1]*dt
    floc=freqs[0] + pfit[2]*df
    floc_e=perr[2]*df

    #Convert Gaussian FWHMs into physical units
    twid=pfit[3]*dt
    twid_e=perr[3]*dt
    fwid=pfit[4]*np.abs(df)
    fwid_e=perr[4]*np.abs(df)

    print "Location of center: %.1f +/- %.2f ms, %.1f +/- %.1f MHz" % (tloc*1e3, tloc_e*1e3, floc, floc_e)
    print "Widths of fit: %.2f +/- %.2f ms, %.1f +/- %.1f MHz" % (twid*1e3, twid_e*1e3, fwid, fwid_e)
    times=np.append(bin_time,tloc)
    times=np.append(times,twid)
    everything=np.array([pfit[0], tloc, tloc_e, floc, floc_e, twid, twid_e, fwid, fwid_e, pfit[5]])
    if tofile:
        np.savetxt(tofile, everything.transpose())

    return times, everything, bin_center

def fit_my_smudge(tfdata, stimes, freqs, guess=[], doplot=True, basename="my_smudge"):
    '''
    tfdata = 2D numpy array. axix 0 is time and axis 1 is freq
    stimes = 1D array contining time of each time sample
    freqs = 1D array contining freq of each channel
    guess = an optional guess for the 2D Gaussian fitting. Otherwise code tries figure it out.
    doplot = Boolean, should it make plot or not?
    basename = basename for filename when saving the plots
    '''
    #Row names for output fits
    AMP,TLOC,UTLOC,FLOC,UFLOC,TWID,UTWID,BW,UBW,OFF=0,1,2,3,4,5,6,7,8,9

    #Fit the data. The returned fit is in *bins*
    pfit,uncert=fit_burst_component(tfdata, maxfev=2800, guess=guess)

    #Convert the fit from bins into physical values
    #Rows: GaussAmp, timeCenter, timeCenter_uncert, freqCenter, freqCenter_uncert, timeWidth, timeWidth_uncert, freqWidth, freqWidth_uncert, baseline
    times, cpfit, bin_center=convert_fit(pfit, uncert, stimes, freqs, tofile='%s.txt'%basename)

    if doplot:
        #Plot fit on top of data
        plot_burst_windows(np.arange(tfdata.shape[1]), np.arange(tfdata.shape[0]), tfdata, cpfit, annotate=True, pfit=pfit) #swap axes
        fig=py.gcf()
        ax=fig.get_axes()
        py.savefig(basename+'_bestfit.png')
        py.show()

        #Plot data residuals
        plot_burst_residuals(np.arange(tfdata.shape[1]), np.arange(tfdata.shape[0]), tfdata, pfit=pfit) #swap axes
        py.gcf()
        py.savefig(basename+'_resid.png')
        py.show()

    return times, cpfit, bin_center

