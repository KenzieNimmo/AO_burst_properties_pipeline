"""
Filterbank of burst -> fluence and peak flux density of the burst
Kenzie Nimmo 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import optparse
from presto import psrfits
import pandas as pd
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

def get_SEFD(za,give_freq):
    """
    za is the zenith angle
    give_freq is the central frequency in MHz from fit
    """
    #find the nearest frequencies with data surrounding the given freq
    
    if za<2 or za>20:
        print("check ZA, out of range, errors on SEFD will likely be large")
    nearest_frq1=min(frqs, key=lambda x:abs(x-give_freq))
    if give_freq>nearest_frq1 and give_freq<1666:
        nearest_frq2=frqs[frqs.index(nearest_frq1)+1]
    elif give_freq<nearest_frq1 and give_freq>1155:
        nearest_frq2=frqs[frqs.index(nearest_frq1)-1]
    elif give_freq>1666:
        print('invalid freq, too high, using 1666 MHz')
        return SEFD(za,1666)
    elif give_freq<1155:
        print('invalid freq, too low, using 1155 MHz')
        return SEFD(za,1155)
        

    #interpolates a SEFD
    wt1=np.linalg.norm(give_freq-nearest_frq1)/np.linalg.norm(nearest_frq1-nearest_frq2)
    wt2=np.linalg.norm(give_freq-nearest_frq2)/np.linalg.norm(nearest_frq1-nearest_frq2)
    vall=np.average([SEFD(za,nearest_frq1),SEFD(za,nearest_frq2)],weights=[wt2,wt1])
    
    return vall
  

def radiometer(tsamp, bw, npol, cntr_freq,za):
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
    SEFD=get_SEFD(cntr_freq,za)
    return (SEFD) * (1 / np.sqrt((bw * 1.e6) * npol * tsamp * 1e-3))


def fluence_flux(arr, bw, t_cent, width, tsamp, SEFD, offpulse):
    # (arr, bw, t_cent, width, width_error, tsamp, SEFD, offpulse):
    """
    fluence_flux(arr, bw, t_cent, width, tsamp, offpulse)
    arr is the burst dynamic spectrum
    bw is the bandwidth in MHz
    t_cent is the peak time of the burst found using the 2D Gaussian fit (or by eye)
    width is the FWHM duration of the burst found using the 2D Gaussian fit (or by eye)
    tsamp is the sampling time of the data in seconds
    offpulse is the pickle file containing the offpulse times
    Idea is to subtract mean and divide by the rms to normalize the time series
    (making the noise ~1 and so the height of the signal is equal to the S/N)
    Then to convert to physical units (Jy ms), we use the radiometer equation.
    Also use same method to determine peak flux in physical units (Jy)
    """

    with open(str(offpulse), 'rb') as f:
        offtimes = pickle.load(f)

    totalbins = arr.shape[1]  # number of bins
    offtimes = offtimes[np.where(offtimes < totalbins)[0]]

    print(offtimes)
    t_cent = t_cent / tsamp  # in bins
    # TODO: t_cent is with reference to the obsrvation start not the cutout
    width = width / tsamp  # in bins

    conv = 2.355
    width = int((width * 2. / conv))
    t_cent = int(t_cent)

    tsamp *= 1e3  # milliseconds

    profile = np.sum(arr, axis=0)
    spec = np.sum(arr[:, (t_cent - width):(t_cent + width)], axis=1)
    offprof = np.sum(arr[:, offtimes], axis=0)
    offspec = np.sum(arr[:, offtimes], axis=1)
    mean = np.mean(offprof)
    meanspec = np.mean(offspec)
    offprof -= mean
    profile -= mean
    spec -= meanspec
    std = np.std(offprof)

    stdspec = np.std(offspec)
    offprof /= std
    profile /= std
    spec /= stdspec

    profile_burst = profile[(t_cent - width):(t_cent + width)]
    spec_burst = spec

    plt.plot(profile, 'k')
    plt.axvline((t_cent - width), color='r')
    plt.axvline((t_cent + width), color='r')

    plt.xlabel('Time bins')
    plt.ylabel('S/N')
    plt.savefig('burst_profile.pdf', format='pdf')
    plt.show()

    fluence = np.sum(profile_burst * radiometer(tsamp, bw, 2, SEFD) * tsamp)  # fluence
    peakSNR = np.max(profile_burst)
    flux = np.max(profile_burst * radiometer(tsamp, bw, 2, SEFD))  # peak flux density
    prof_flux = profile * radiometer(tsamp, bw, 2, SEFD)
    spec_flux = spec_burst * radiometer(tsamp, bw, 2, SEFD)

    # assuming 20% error on SEFD dominates, even if you consider the errors on
    # width and add them in quadrature i.e.
    # sigma_flux**2+sigma_width**2=sigma_fluence**2, sigma_fluence~0.2
    fluence_error = 0.2 * fluence

    # error_bin=(width_error/len(profile_burst))
    # errors=[]
    # for i in range(len(profile_burst)):
    #    error_box=np.abs(profile_burst[i]*radiometer(tsamp,bw,2,SEFD)*tsamp)*np.sqrt((0.2)**2+(error_bin)**2)
    #    errors=np.append(errors,error_box)

    # x=0
    # for i in range(len(errors)):
    #    x+=errors[i]**2

    # fluence_error=np.sqrt(x)

    return fluence, flux, prof_flux, spec_flux, peakSNR, fluence_error


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
    parser = optparse.OptionParser(usage='%prog [options] infile',
                                   description="Fluence and peak flux density of FRB. Input the "
                                               "pickle file output from fit_burst_fb.py.")
    parser.add_option('-S', '--SEFD', dest='SEFD', type='float',
                      help="System Equivalent Flux Density [Jy].", default=None)
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

    smooth = None  # smoothing value used for bandpass calibration

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')

    # Get pulses to be processed.
    if options.pulse is not None:
        burst_ids = [options.pulse]
    else:
        burst_ids = pulses.index.get_level_values(0).unique()
        if 'Peak S/N' in pulses.columns:
            not_processed = pulses.loc[(slice(None), 'sb1'), 'Peak S/N"'].isna()
            burst_ids = burst_ids[not_processed]

    for burst_id in burst_ids:
        print("Fluence/Peak Flux Density of observation %s, pulse ID %s" %(basename, burst_id))
        burst_dir = burst_id
        filename = f'{burst_dir}/{basename}_{burst_id}.fits'

        # read in mask file
        maskfile = f'{burst_dir}/{basename}_{burst_id}_mask.pkl'
        # read in offpulse file
        offpulsefile = f'{burst_dir}/{basename}_{burst_id}_offpulse_time.pkl'

        # Get observation parameters
        fits = psrfits.PsrfitsFile(filename)
        tsamp = fits.specinfo.dt
        freqs = fits.frequencies
        nchan = fits.specinfo.num_channels
        fres = fits.specinfo.df
        bw = fits.specinfo.BW

        t_cent = pulses.loc[burst_id, 't_cent / s']
        t_std = pulses.loc[burst_id, 't_std / s']
        dm = pulses.loc[(burst_id, 'sb1'), 'DM']

        waterfall, t, peak_bin = import_fil_fits.fits_to_np(
            filename, dm=dm, maskfile=maskfile, bandpass=True, offpulse=offpulsefile, AO=True,
            smooth_val=smooth, hdf5=orig_in_hdf5_file, index=burst_id)

        if options.SEFD is None:
            if np.max(freqs) < 2000.:
                print("Data is assumed to be L-band")
                SEFD = 30 / 10.
            if np.max(freqs) >= 2000.:
                print("Data is assumed to be C-band")
                SEFD = 28. / 6.
        else:
            SEFD = options.SEFD

        for sb in pulses.loc[burst_id].index:
            fluence, flux, prof_flux, spec_flux, peakSNR, fluence_error = fluence_flux(
                waterfall, bw, t_cent.loc[sb], t_std.loc[sb], tsamp, SEFD, offpulsefile)

            print("Peak S/N", peakSNR)
            print("Fluence:", fluence, "+-", fluence_error, "Jy ms")
            print("Peak Flux Density:", flux, "Jy")

            pulses.loc[(burst_id, sb), 'S/N Peak'] = peakSNR
            pulses.loc[(burst_id, sb), 'Fluence / Jy ms'] = fluence
            pulses.loc[(burst_id, sb), 'Fluence error / Jy ms'] = fluence_error
            pulses.loc[(burst_id, sb), 'Peak Flux Density / Jy'] = flux

            if options.distance is not None:
                specenerg = energy_iso(fluence, options.distance)
                print("Spectral energy density:", specenerg, r"erg Hz^-1")
                pulses.loc[(burst_id, sb), 'Spectral Energy Density / erg Hz^-1'] = specenerg
                pulses.loc[(burst_id, sb), 'Distance / Mpc'] = options.distance

    pulses.to_hdf(out_hdf5_file, 'pulses')

