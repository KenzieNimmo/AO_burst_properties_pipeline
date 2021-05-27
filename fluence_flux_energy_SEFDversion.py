"""
Filterbank of burst -> fluence and peak flux density of the burst
Kenzie Nimmo 2020
"""
import numpy as np
import sys
import os
import optparse
import pickle
import pandas as pd
import warnings
from astropy import time, coordinates as coord, units as u #, constants as const

from presto import psrfits

import import_fil_fits
#hardcoded fit values for SEFD at different frequencies
popt1155=np.array([  0.02744894,   1.11956666,   2.56991623, -16.95838086, 4.52554854])
popt1310=np.array([  0.03635505,   1.13631723,   2.14199018, -12.78659945, 3.63049544])
popt1375=np.array([  0.03035996,   1.15606075,   2.03391971, -12.01735757, 3.50332863])
popt1415=np.array([  0.18230778,   1.11305479,   2.90374053, -36.41706102, 3.36719088])
popt1550=np.array([  0.03052548,   1.15298379,   2.20894805, -15.13983849, 3.31135101])
popt1610=np.array([  0.12903002,   1.13016458,   3.19272939, -41.49687011, 3.21341672])
popt1666=np.array([ 1.30414741e-02,  1.16920252e+00,  2.57425057e+00, -1.92565818e+01, 3.19924629e+00])

frqs=[1155,1310,1375,1415,1550,1610,1666]

#exponential function, other options might be better, but this is sufficient for our purposes
#this likely *very* slightly underestimates SEFD at low za an overestimates at high za
def func(x, a, b, c, d, e):

    return  a*b**(c*x+d)+e

#endless elifs
def SEFD(za, cf):
    """
    za is the zenith angle
    cf is the central frequency in MHz, has to be in (1155,1310,1375,1415,1550,1610,1666)
    """
    if cf==1155:
        SEFD=np.round(func(za,*popt1155),2)
    elif cf==1310:
        SEFD=np.round(func(za,*popt1310),2)
    elif cf==1375:
        SEFD=np.round(func(za,*popt1375),2)
    elif cf==1415:
        SEFD=np.round(func(za,*popt1415),2)
    elif cf==1550:
        SEFD=np.round(func(za,*popt1550),2)
    elif cf==1610:
        SEFD=np.round(func(za,*popt1610),2)
    elif cf==1666:
        SEFD=np.round(func(za,*popt1666),2)
    else:
        print("Incorrect central frequency provided. Choose from 1155, 1310, 1375, 1415, 1550, 1610, 1666 MHz")
    return SEFD


def get_SEFD(za, give_freq):
    """
    za is the zenith angle
    give_freq is the central frequency in MHz from fit
    """
    #find the nearest frequencies with data surrounding the given freq

    if za < 2 or za > 20:
        print("check ZA, out of range, errors on SEFD will likely be large")
    nearest_frq1 = min(frqs, key=lambda x: abs(x-give_freq))
    if give_freq >= nearest_frq1 and give_freq < 1666:
        nearest_frq2 = frqs[frqs.index(nearest_frq1)+1]
    elif give_freq < nearest_frq1 and give_freq >= 1155:
        nearest_frq2 = frqs[frqs.index(nearest_frq1)-1]
    elif give_freq > 1666:
        #print('invalid freq, too high, using 1666 MHz')
        return SEFD(za, 1666)
    elif give_freq < 1155:
        #print('invalid freq, too low, using 1155 MHz')
        return SEFD(za, 1155)

    #interpolates a SEFD
    wt1 = np.linalg.norm(give_freq-nearest_frq1) / np.linalg.norm(nearest_frq1-nearest_frq2)
    wt2 = np.linalg.norm(give_freq-nearest_frq2) / np.linalg.norm(nearest_frq1-nearest_frq2)
    vall = np.average([SEFD(za, nearest_frq1), SEFD(za, nearest_frq2)], weights=[wt2, wt1])

    return vall


def radiometer(tsamp, bw, npol, cntr_freq, za):
    """
    radiometer(tsamp, bw, npol, Tsys, G):
    tsamp is the time resolution in milliseconds
    bw is the bandwidth in MHz
    npol is the number of polarizations
    cntr_freq is the center frequency of the Gaussian fit in MHz
    za is the zenith angle
    Tsys is the system temperature in K (typical value for Effelsberg = 20K)
    G is the telescope gain in K/Jy (typical value for Effelsberg = 1.54K/Jy)
    """
    if isinstance(cntr_freq, (list, tuple, np.ndarray)):
        SEFD = np.array([get_SEFD(za, cntr_f) for cntr_f in cntr_freq])
    else:
        SEFD = get_SEFD(za, cntr_freq)
    return SEFD / np.sqrt(npol * (bw * 1.e6) * tsamp * 1e-3)


def calc_fluence(waterfall, chan_bw, tsamp, chan_freqs, za):
    """Calculate the fluence, flux, and peak S/N.
    To convert to physical units (Jy ms), we use empiric functions found for AO.
    We also use same method to determine peak flux in physical units (Jy).
    waterfall : The burst dynamic spectrum in S/N units, but only the window that should be used
        for the calculations.
    bw : The bandwidth of the channels in MHz
    tsamp : The sampling time of the data in seconds
    chan_freqs : Array with the channel center frequencies.
    za : Zenith Angle at the time of the Burst.
    """
    tsamp *= 1e3  # milliseconds

    spectrum = np.sum(waterfall, axis=1) * tsamp

    chan_radiometer = radiometer(tsamp, chan_bw, 2, chan_freqs, za)

    fluence = np.mean(spectrum * chan_radiometer)

    # dunno how called
    for_energy = np.mean(spectrum * chan_radiometer * chan_freqs * 1e6) #to Hz

    return fluence, for_energy


def fluence_estimate(waterfall, chan_bw, tsamp, central_freq, za, offpulsefile,
                 window_start, window_end):
    """Calculate the fluence, flux, and peak S/N.

    To convert to physical units (Jy ms), we use empiric functions found for AO.
    We also use same method to determine peak flux in physical units (Jy).

    waterfall : The burst dynamic spectrum in S/N units, but only the window that should be used
        for the calculations.
    bw : The bandwidth of the channels in MHz
    tsamp : The sampling time of the data in seconds
    chan_freqs : Array with the channel center frequencies.
    za : Zenith Angle at the time of the Burst.
    """
    tsamp *= 1e3  # milliseconds

    offpulse = pickle.load(open(offpulsefile, 'rb'))

    time_series = np.sum(waterfall, axis=0)
    time_series -= time_series[offpulse].mean()
    time_series /= time_series[offpulse].std()
    time_series = time_series[window_start:window_end]

    bw = chan_bw * np.sum(~waterfall.mask[:, window_start])
    chan_radiometer = radiometer(tsamp, bw, 2, central_freq, za)

    fluence = time_series.sum() * radiometer(tsamp, bw, 2, central_freq, za) * tsamp

    return fluence


def energy_iso(fluence, distance_lum):
    """Following Law et al. (2017)
    fluence in Jy ms
    distance_lum in Mpc
    At the moment bw not needed, units are therefore erg Hz^-1
    """
    # convert Jy ms to J s
    fluence_Jys = fluence * 1e-3
    # convert Mpc to cm
    distance_lum_cm = 3.086e24 * distance_lum
    return fluence_Jys * 4 * np.pi * (distance_lum_cm**2) * 1e-23


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Polarization is AABBCRCI, averaging AA and BB")

    parser = optparse.OptionParser(usage='%prog [options] infile',
                                   description="Fluence and peak flux density of FRB. Input the "
                                               "pickle file output from fit_burst_fb.py.")
    parser.add_option('-d', '--distance', dest='distance', type='float',
                      help="Distance to the FRB for energy calculation in Mpc (not required).",
                      default=None)
    parser.add_option('-p', '--pulse', type='str', default=None,
                      help="Give a pulse id to process only this pulse.")
    parser.add_option('-r', '--reproc', default=False, action='store_true',
                      help="Reprocess all bursts.")

    options, args = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    elif len(args) != 1:
        sys.stderr.write("Only one input file must be provided!\n")
    else:
        options.infile = args[-1]

    # getting the basename of the observation and pulse IDs
    basename = options.infile
    # hdf5 file
    orig_in_hdf5_file = f'{basename}.hdf5'
    in_hdf5_file = f'{basename}_burst_properties.hdf5'
    out_hdf5_file = in_hdf5_file

    if not os.path.isfile(orig_in_hdf5_file) and os.path.isdir(basename):
        os.chdir(os.path.join(basename, 'pulses'))

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')

    # Get pulses to be processed.
    if options.pulse is not None:
        burst_ids = [options.pulse]
    else:
        burst_ids = pulses.index.get_level_values(0).unique()
        if not options.reproc and ('General', 'Fluence / Jy ms') in pulses.columns:
            not_processed = pulses.loc[(slice(None), 'sb1'),  ('General', 'Fluence / Jy ms')].isna()
            burst_ids = burst_ids[not_processed]

    for burst_id in burst_ids:
        print("Fluence/Peak Flux Density of observation %s, pulse ID %s" %(basename, burst_id))
        base_pulse = burst_id.split('-')[0]
        filename = f'{base_pulse}/{basename}_{base_pulse}.fits'

        # read in mask file
        maskfile = f'{base_pulse}/{basename}_{base_pulse}_mask.pkl'
        # read in offpulse file
        offpulsefile = f'{base_pulse}/{basename}_{base_pulse}_offpulse_time.pkl'

        # Get observation parameters
        fits = psrfits.PsrfitsFile(filename)
        tsamp = fits.specinfo.dt
        freqs = fits.frequencies
        nchan = fits.specinfo.num_channels
        chan_bw = fits.specinfo.df

        time_guesses = pulses.loc[burst_id, ('Guesses', 't_cent')]
        freq_peak_guess = pulses.loc[burst_id, ('Guesses', 'f_cent')]
        tfit = pulses.loc[(burst_id, 'sb1'), ('General', 'tfit')]
        dm = pulses.loc[(burst_id, 'sb1'), ('General', 'DM')]

        waterfall, t, peak_bin = import_fil_fits.fits_to_np(
            filename, dm=dm, maskfile=maskfile, AO=True,
            hdf5=orig_in_hdf5_file, index=base_pulse)
        waterfall = import_fil_fits.bandpass_calibration(waterfall, offpulsefile, tavg=1,
                                                         AO=True, smooth_val=None, plot=False)

        # Take only the window around the burst that was used for fitting cause it's safe
        # from dropouts.
        window_start = int(time_guesses.min() - tfit*1e-3/tsamp)
        widow_end = int(time_guesses.max() + tfit*1e-3/tsamp)

        # Calculate the zenith angle because AOs sensitivty depends on it.
        # 20 is arbitrary but presto only gives the zenith angle of the first subint
        # (although the rest is in the fits file as well).
        if fits.nsubints < 20:
            za = fits.specinfo.zenith_ang
        else:
            arecibo_coord = coord.EarthLocation.from_geodetic(
                lon=-(66. + 45./60. + 11.1/3600.)*u.deg,  # arecibo geodetic coords in deg
                lat=(18. + 20./60. + 36.6/3600.)*u.deg,
                height=497.*u.m)

            file_tstart = fits.specinfo.start_MJD[0]
            guess_mjd = file_tstart + time_guesses.mean()*tsamp / (24 * 3600)
            guess_mjd = time.Time(guess_mjd, format='mjd', scale='utc', location=arecibo_coord)

            R1_coord = coord.SkyCoord(fits.specinfo.ra_str, fits.specinfo.dec_str,
                                      unit=(u.hourangle, u.deg), frame='icrs', obstime=guess_mjd)
            altaz = R1_coord.transform_to(coord.AltAz(location=arecibo_coord))
            za = 90 - altaz.alt.deg

        # Put it all into that function. , flux, peakSNR, fluence_error
        fluence_est = fluence_estimate(waterfall, chan_bw, tsamp, freq_peak_guess.mean(), za,
                                       offpulsefile, window_start, widow_end)

        waterfall = waterfall[:, window_start:widow_end]
        fluence, for_energy = calc_fluence(waterfall, chan_bw, tsamp, freqs, za)

        print("Fluence:", fluence, "Jy ms")
        print("Fluence estimate:", fluence_est, "Jy ms")

        pulses.loc[burst_id, ('General', 'Fluence / Jy ms')] = fluence
        pulses.loc[burst_id, ('General', 'Fluence estimate / Jy ms')] = fluence_est

        if options.distance is not None:
            specenerg = energy_iso(fluence, options.distance)
            energy = energy_iso(for_energy, options.distance)
            print("Spectral energy density:", specenerg, r"erg Hz^-1")
            pulses.loc[burst_id,  ('General', 'Spectral Energy Density / erg Hz^-1')] = specenerg
            pulses.loc[burst_id,  ('General', 'Energy / erg')] = energy
            pulses.loc[burst_id,  ('General', 'Distance / Mpc')] = options.distance

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            pulses.to_hdf(out_hdf5_file, 'pulses')

