"""
Makes an array (masked or not) from a filterbank or fits file. Also applies bandpass correction
using an offpulse region.
"""

#import fb_pipe as filterbank
from presto import filterbank
import numpy as np
import pickle
#from scipy.interpolate import interp1d as interp
import matplotlib.pyplot as plt
#import psrfits_pipe as psrfits
from presto import psrfits
from presto import psr_utils
#from astropy.io import fits as astrofits
import pandas as pd
import sys

from astropy import units as u, constants as const


def filterbank_to_np(filename, dm=None, maskfile=None,
                     bandpass=False, offpulse=None, smooth_val=None):
    """
    Read filterbank file and output a numpy array of the data
    To dedisperse, give a dm value.
    To mask, give a maskfile (a pkl file containing the channels to mask)
    To bandpass correct, bandpass=True, give offpulse file (a pkl file containing the
    offpulse time bins) and give smooth_val (an integer) to define how much smoothing
    the bandpass should have (None for no smoothing).
    """

    fil = filterbank.filterbank(filename)
    total_N = fil.number_of_samples
    spec = fil.get_spectra(0, total_N)
    if dm is not None:
        spec.dedisperse(dm, padval='mean')
    arr = np.array([spec[i] for i in range(fil.header['nchans'])])
    t_samp = fil.header['tsamp']
    if maskfile is not None:
        amaskfile = pickle.load(open(maskfile, 'rb'))
        amask = [int(i) for i in amaskfile]
        vmin = np.amin(arr)
        arr[amask, :] = vmin - 100
        mask = arr < vmin - 50
        arr = np.ma.masked_where(mask, arr)
    arr = np.flip(arr, 0)
    #if bandpass is True and offpulse is not None:
    #    arr = bp(filename, maskfile, nbins, offpulse, smooth_val=smooth_val)
    return arr


def fits_to_np(filename, dm=None, maskfile=None, bandpass=False, smooth_val=None,
               AO=False, hdf5=None, index=None, plot=False, tavg=1, t_cut=200e-3):
    """
    Read psrfits file and output a numpy array of the data
    To dedisperse, give a dm value.
    To mask, give a maskfile (a pkl file containing the channels to mask)
    To bandpass correct, bandpass=True, give offpulse file (a pkl file
    containing the offpulse time bins) and give smooth_val (an integer) to
    define how much smoothing the bandpass should have (None for no smoothing).

    For AO FRB 121102 analysis, AO=True. AO chopped fits files contain two
    subints before the burst and 8 after, so we use this to chop out a smaller
    chunk of data around the burst for processing (also to ensure each burst
    is processed consistently).
    If AO==True give the hdf5 file containing the burst information to use.
    """
    fits = psrfits.PsrfitsFile(filename)
    total_N = int(fits.specinfo.N)
    t_samp = fits.specinfo.dt
    #npoln = fits.npoln
    #imjd, fmjd = psrfits.DATEOBS_to_MJD(fits.specinfo.date_obs)
    #tstart = imjd + fmjd
    if AO is True:
        #fits_file = astrofits.open(filename, memmap=True)
        #subint_hdu = fits_file[1]
        #subint_hdr = subint_hdu.header
        #begin_subint = subint_hdr['NSUBOFFS']
        #sample_per_subint = subint_hdr['NSBLK']
        #time_from_orig_begin_time = (
        #    begin_subint * sample_per_subint * t_samp) / (24 * 3600.)  # MJD
        #new_tstart = tstart + time_from_orig_begin_time
        imjd, fmjd = psrfits.DATEOBS_to_MJD(fits.specinfo.date_obs)
        obs_start = imjd + fmjd                  # The observation start in MJD
        file_tstart = fits.specinfo.start_MJD[0]  # The cutout start
        # In seconds after the observation starts
        file_tstart = (file_tstart - obs_start) * 24 * 3600.

        if hdf5 is not None and index is not None:
            hdf5_file = hdf5
            if isinstance(hdf5_file, pd.DataFrame):
                pulses = hdf5_file
            else:
                pulses = pd.read_hdf(hdf5_file, 'pulses')
            index = int(index)
            pulses = pulses.loc[pulses['Pulse'] == 0]
            burst_time = pulses.loc[index, 'Time'] #/ (24 * 3600.) + obs_start  # MJD

            # NB: There is an offset between the burst peak time determined above and the burst by ~4.5ms.
            # I think this is a dedispersion artefact but not sure. At the moment hard-coding a shift
            # burst_time-=(4.5e-3/(24*3600.)) #MJD
            # I think the new presto version does this correctly.

        else:
            print("Please provide the hdf5 file containing the burst properties and the burst \
                  index from the search pipeline.")

        spec = fits.get_spectra(0, total_N - 1)
    else:
        spec = fits.get_spectra(0, total_N - 1)

    if dm is not None:
        # Presto uses a dm_const rounded to four significant digits, so we calculate the shifts
        # ourself.
        dm_const = (const.e.gauss**2/(2*np.pi*const.m_e*const.c)).to(u.cm**3/u.pc*u.MHz**2*u.s)
        f_top = fits.specinfo.hi_freq
        freqs = spec.freqs
        dm = float(dm)
        shifts = np.round((dm_const.value*(1./(freqs)**2 - 1./(f_top)**2)*dm)/t_samp).astype(np.int)
        spec.shift_channels(shifts)

        #spec.dedisperse(dm, padval='mean')
        # Trim the spectrum to get rid of padded values
        #f_bottom = fits.specinfo.lo_freq
        #max_shift = int(round((psr_utils.delay_from_DM(dm, f_bottom)
        #                       - psr_utils.delay_from_DM(dm, f_top)) / t_samp))
        max_shift = shifts[-1]
        spec.trim(max_shift)

    arr = spec.data

    if smooth_val == 1:
        smooth_val = None

    if tavg > 1:
        nsamples = arr.shape[1]
        if nsamples % tavg != 0:
            #print("The cutout is slightly adjusted to fit the downsample factor.")
            arr = arr[:, : -(nsamples % tavg)]

        newsamps = nsamples // tavg
        arr = arr.reshape(arr.shape[0], newsamps, tavg).mean(axis=-1)
        t_samp *= tavg

    if maskfile is not None:
        amaskfile = pickle.load(open(maskfile, 'rb'))
        mask = np.zeros(arr.shape, dtype=np.bool)
        mask[amaskfile] = True
        arr = np.ma.masked_where(mask, arr)

    if AO is True:
        # time in seconds of burst in cropped fits fileim
        burstt = (burst_time - file_tstart)
        peak_bin = int(burstt / t_samp)

        # Make the cutout devidable by 64 to avoid conflicts when downsampling later, this assumes
        # a downsampling factor of a power of 2.
        samp_cut = t_cut/2/t_samp
        samp_cut -= samp_cut % (32//tavg)

        # Cut the data around the peak
        start_samp = peak_bin - int(samp_cut)
        end_samp = peak_bin + int(samp_cut)
        arr = arr[:, start_samp : end_samp]
        # Note the number of samples before the new start is start_samp (incl the 0)
        file_tstart += start_samp*t_samp
        return arr, file_tstart, arr.shape[1] // 2
    else:
        return arr


def smooth(x, window_len=11, window='hanning'):
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
    NOTE: length(output) != length(input), to correct this: return
    y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def bandpass_calibration(arr, offpulsefile, tavg=1, AO=False, smooth_val=None, plot=False):
    """
    Uses off pulse data (identified using interactive RFI_zapper.py) to normalise the spectrum and
    calibrate the bandpass
    """
    if smooth_val is not None:
        if smooth_val % 2 == 0:
            print("Please give an ODD smoothing value (2n+1) for the bandpass calibration")
            sys.exit()

    offpulse = pickle.load(open(offpulsefile, 'rb'))
    if tavg != 1:
        # Transform the offtimes into the downsampled shape.
        offlen = offpulse.shape[0]
        offpulse = offpulse.reshape(offlen//tavg, tavg).all(axis=1)
    #offpulse = [int(x) for x in offpulse]
    spec = np.mean(arr[:, offpulse], axis=1)
    chan_std = np.std(arr[:, offpulse], axis=1)
    unsmoothed_spec = spec.copy()
    # smooth the bandpass spectrum
    #speclen = len(spec)

    #mask = np.zeros_like(spec)
    #if maskfile is not None:
    #    amaskfile = pickle.load(open(maskfile, 'rb'))
    #    amask = [speclen - int(i) - 1 for i in amaskfile]
    #    mask[amask] = 1
    if smooth_val is not None:
        print("Caution: smoothing is not currently implemented.")
        print("Not smoothing the bandpass")
        #a = int((smooth_val - 1) / 2.)
        #spec = np.ma.masked_where(mask==True,spec)
        #spec = smooth(spec,window_len=smooth_val)[a-1:-(a+1)]
        #spec = np.ma.masked_where(mask==True,spec)

    #arr2 = arr.copy()

    arr -= spec[:, np.newaxis]
    #for i in range(arr.shape[1]):
    #    arr2[:, i] /= spec
        # arr[:,i]/=maskfit

    arr /= chan_std[:, np.newaxis]

    # diagnostic plots
    if plot is True:
        fig, axes = plt.subplots(2, figsize=[8, 8], sharex=True)
        axes[0].plot(unsmoothed_spec, 'k', label='unsmoothed spectrum')
        axes[0].plot(spec, 'r', label='smoothed spectrum')
        axes[0].legend()

        axes[1].plot(np.mean(arr, axis=1), 'r', label='bandpass corrected')
        axes[1].set_xlabel('Frequency')
        axes[1].legend()
        plt.savefig('bandpass_diagnostic.pdf', format='pdf')
        plt.show()
    return arr
