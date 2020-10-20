"""
Makes an array (masked or not) from a filterbank or fits file. Also applies bandpass correction using an offpulse region.
"""

import filterbank
import numpy as np
import pickle
from scipy.interpolate import interp1d as interp
import matplotlib.pyplot as plt
import psrfits_pipe as psrfits
from astropy.io import fits as astrofits
import pandas as pd

def filterbank_to_np(filename, dm=None, maskfile=None, bandpass=False, offpulse=None, smooth_val=None):
    """
    Read filterbank file and output a numpy array of the data
    To dedisperse, give a dm value.
    To mask, give a maskfile (a pkl file containing the channels to mask)
    To bandpass correct, bandpass=True, give offpulse file (a pkl file containing the offpulse time bins)
    and give smooth_val (an integer) to define how much smoothing the bandpass should have (None for no smoothing).
    """


    fil = filterbank.filterbank(filename)
    total_N = fil.number_of_samples
    spec=fil.get_spectra(0,total_N)
    if dm!=None:
        spec.dedisperse(dm, padval='mean')
    arr = np.array([spec[i] for i in xrange(fil.header['nchans'])])
    t_samp = fil.header['tsamp']
    if maskfile!=None:
        amaskfile = pickle.load(open(maskfile,'rb'))
        amask=[int(i) for i in amaskfile]
        vmin = np.amin(arr)
        arr[amask,:]=vmin-100
        mask = arr<vmin-50
        arr = np.ma.masked_where(mask==True,arr)
    arr=np.flip(arr,0)
    if bandpass==True and offpulse!=None:
        arr=bp(filename,maskfile,nbins,offpulse,smooth_val=smooth_val)
    return arr

def fits_to_np(filename, dm=None, maskfile=None, bandpass=False, offpulse=None,smooth_val=None,AO=False,hdf5=None,index=None):
    """
    Read psrfits file and output a numpy array of the data
    To dedisperse, give a dm value.
    To mask, give a maskfile (a pkl file containing the channels to mask)
    To bandpass correct, bandpass=True, give offpulse file (a pkl file containing the offpulse time bins)
    and give smooth_val (an integer) to define how much smoothing the bandpass should have (None for no smoothing).

    For AO FRB 121102 analysis, AO=True. AO chopped fits files contain two subints before the burst and 8 after,
    so we use this to chop out a smaller chunk of data around the burst for processing (also to ensure each burst is processed consistently).
    If AO==True give the hdf5 file containing the burst information to use.
    """
    fits=psrfits.PsrfitsFile(filename)
    total_N=fits.specinfo.N
    t_samp=fits.specinfo.dt
    npoln = fits.npoln
    imjd,fmjd=psrfits.DATEOBS_to_MJD(fits.specinfo.date_obs)
    tstart=imjd+fmjd
    if AO==True:
        fits_file=astrofits.open(filename,memmap=True)
        subint_hdu=fits_file[1]
        subint_hdr=subint_hdu.header
        begin_subint=subint_hdr['NSUBOFFS']
        sample_per_subint=subint_hdr['NSBLK']
        time_from_orig_begin_time=(begin_subint*sample_per_subint*t_samp)/(24*3600.) #MJD
        new_tstart=tstart+time_from_orig_begin_time
        if hdf5!=None and index!=None:
            hdf5_file = hdf5
            index = int(index)
            pulses = pd.read_hdf(hdf5_file, 'pulses')
            pulses=pulses.loc[pulses['Pulse'] == 0]
            burst_time=pulses.loc[index,'Time']/(24*3600.)+tstart #MJD
        else: print("Please provide the hdf5 file containing the burst properties and the burst index from the search pipeline.")

        burstt=(burst_time-new_tstart)*24*3600. #time in seconds of burst in cropped fits fileim
        peak_bin = burstt/t_samp

        begin_bin=0 #number of bins
        peak_subint=int(np.floor(peak_bin/sample_per_subint))
        end_bin=int(peak_subint+6.)*sample_per_subint

        spec=fits.get_spectra(begin_bin,end_bin)

    else: spec=fits.get_spectra(0,total_N)

    if dm!=None:
        spec.dedisperse(dm, padval='mean')


    arr = np.array([spec[i] for i in xrange(fits.specinfo.num_channels)])

    if maskfile!=None:
        amaskfile = pickle.load(open(maskfile,'rb'))
        amask=[int(i) for i in amaskfile]
        vmin = np.amin(arr)
        arr[amask,:]=vmin-100
        mask = arr<vmin-50
        arr = np.ma.masked_where(mask==True,arr)
    arr=np.flip(arr,0)
    if smooth_val==1:
        smooth_val=None
    if bandpass==True and offpulse!=None:
        arr=bp(arr,maskfile,offpulse,AO=AO,smooth_val=smooth_val)
    if AO==True:
        return arr, new_tstart, peak_bin
    else:
        return arr

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def bp(arr,maskfile,offpulsefile,AO=False,smooth_val=None):
    """
    Uses off pulse data (identified using interactive RFI_zapper.py) to normalise the spectrum and
    calibrate the bandpass
    """

    offpulse=pickle.load(open(offpulsefile,'rb'))
    spec = np.mean(arr[:,offpulse],axis=1)
    unsmoothed_spec = spec.copy()
    # smooth the bandpass spectrum
    speclen=len(spec)

    mask=np.zeros_like(spec)
    if maskfile!=None:
        amaskfile = pickle.load(open(maskfile,'rb'))
        amask=[speclen-int(i)-1 for i in amaskfile]
        mask[amask]=1
    if smooth_val!=None:
        spec = smooth(spec,window_len=smooth_val)[:speclen]
        spec = np.ma.masked_where(mask==True,spec)

    arr2=arr.copy()

    for i in range(arr.shape[1]):
        arr2[:,i]/=spec
        #arr[:,i]/=maskfit

    arr2-=np.mean(arr2)

    #diagnostic plots
    fig,axes=plt.subplots(2,figsize=[8,8],sharex=True)
    axes[0].plot(unsmoothed_spec,'k',label='unsmoothed spectrum')
    axes[0].plot(spec,'r',label='smoothed spectrum')
    axes[0].legend()

    axes[1].plot(np.mean(arr2,axis=1),'r',label='bandpass corrected')
    axes[1].set_xlabel('Frequency')
    axes[1].legend()
    plt.savefig('bandpass_diagnostic.pdf',format='pdf')
    plt.show()
    return arr2
