"""
Filterbank of burst -> fluence and peak flux density of the burst
Kenzie Nimmo 2020
"""
import numpy as np
import sys
import os
import optparse
from presto import psrfits
import pandas as pd
import warnings
from astropy import time, coordinates as coord, units as u #, constants as const
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


def radiometer(tsamp, bw, npol, cntr_freqs, za):
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
    SEFD = np.array([get_SEFD(za, cntr_freq) for cntr_freq in cntr_freqs])
    return (SEFD) * (1 / np.sqrt((bw * 1.e6) * npol * tsamp * 1e-3))


def fluence_flux(waterfall, bw, tsamp, chan_freqs, za):
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

    profile_burst = np.sum(waterfall, axis=0)
    #spec_burst = np.sum(waterfall, axis=1)

    chan_radiometer = radiometer(tsamp, bw, 2, chan_freqs, za)

    #prof_flux = np.sum(waterfall * chan_radiometer[:, np.newaxis], axis=0)

    fluence = np.sum(waterfall * chan_radiometer[:, np.newaxis]) * tsamp
    #peakSNR = np.max(profile_burst)
    #flux = np.max(prof_flux)  # peak flux density

    # assuming 20% error on SEFD dominates, even if you consider the errors on
    # width and add them in quadrature i.e.
    # sigma_flux**2+sigma_width**2=sigma_fluence**2, sigma_fluence~0.2
    #fluence_error = 0.2 * fluence

    return fluence #, flux, peakSNR, fluence_error


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
        if ('General', 'Fluence / Jy ms') in pulses.columns:
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
        bw = fits.specinfo.BW

        time_guesses = pulses.loc[burst_id, ('Guesses', 't_cent')]
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

        waterfall = waterfall[:, window_start:widow_end]

        # Calculate the zenith angle because AOs sensitivty depends on it.
        # 20 is arbitrary but presto only gives the zenith angle of the first subint.
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
        fluence = fluence_flux(
                waterfall, bw=chan_bw, tsamp=tsamp, chan_freqs=freqs, za=za)

        #for sb in pulses.loc[burst_id].index:

        #print("Peak S/N", peakSNR)
        print("Fluence:", fluence) #, "+-", fluence_error, "Jy ms")
        #print("Peak Flux Density:", flux, "Jy")

        #pulses.loc[burst_id, ('General', 'S/N Peak')] = peakSNR
        pulses.loc[burst_id, ('General', 'Fluence / Jy ms')] = fluence
        #pulses.loc[burst_id, ('General', 'Fluence error / Jy ms')] = fluence_error
        #pulses.loc[burst_id, ('General', 'Peak Flux Density / Jy')] = flux

        if options.distance is not None:
            specenerg = energy_iso(fluence, options.distance)
            print("Spectral energy density:", specenerg, r"erg Hz^-1")
            pulses.loc[burst_id,  ('General', 'Spectral Energy Density / erg Hz^-1')] = specenerg
            pulses.loc[burst_id,  ('General', 'Distance / Mpc')] = options.distance

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            pulses.to_hdf(out_hdf5_file, 'pulses')

