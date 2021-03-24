#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:22:47 2021

@author: jjahns
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import import_fil_fits

from astropy.modeling import models, fitting
from presto import psrfits


def fit_multiple_1d_gaussians(x_data, y_data, amp_guess, x_guess, std_guess, weights=None):
    # Create the initial model.
    g_guess = models.Gaussian1D(amplitude=amp_guess[0], mean=x_guess[0], stddev=std_guess[0])
    for i in range(1, len(amp_guess)):
        g_guess += models.Gaussian1D(amplitude=amp_guess[i], mean=x_guess[i], stddev=std_guess[i])

    # Fit the model
    fitter = fitting.LevMarLSQFitter()
    gaussians = fitter(g_guess, x_data, y_data, weights=weights, maxiter=500)

    return gaussians, fitter


def fit_time_and_freq(waterfall, tsamp, freqs, offpulsefile, amp_guess, t_guess, t_std_g, f_guess,
                      f_std_g, t_fit, downs, make_plot=False):
    """
    waterfall : waterfall bandpass corrected to SNR
    """

    # Give default values if t_std and f_std are not given
    t_std_g[t_std_g.isna()] = 1e-3  # 1ms
    f_std_g[f_std_g.isna()] = 100   # 100 MHz

    offpulse = pickle.load(open(offpulsefile, 'rb'))
    if downs != 1:
        # Transform the offtimes into the downsampled shape.
        offlen = offpulse.shape[0]
        offpulse = offpulse.reshape(offlen//downs, downs).all(axis=1)

    # Collapse to time series and renormalize to rms.
    time_series = waterfall.mean(axis=0)
    off_mean = time_series[offpulse].mean()
    off_std = time_series[offpulse].std()
    time_series = (time_series - off_mean) / off_std

    # Get the widow to be fitted.
    window_start = int(t_guess.min() - t_fit*1e-3/tsamp)
    widow_end = int(t_guess.max() + t_fit*1e-3/tsamp)

    times = np.arange(time_series.shape[0]) * tsamp
    time_series = time_series[window_start:widow_end]

    # Convert t_guess to time units and lower amp_guess
    t_guess = t_guess * tsamp
    amp_guess = amp_guess / off_std / 2  # assume a pulse goes over half the band

    gaussians, fitter = fit_multiple_1d_gaussians(times[window_start:widow_end], time_series,
                                                  amp_guess, t_guess, t_std_g)

    gauss_params = gaussians.parameters.reshape((gaussians.n_submodels, 3))
    cov_mat = fitter.fit_info['param_cov']

    if cov_mat is None:
        print("No covariance matrix was returned. " + fitter.fit_info['message'])
        uncert = np.full(gauss_params.size, np.nan)
    else:
        print(f"The fit converged after {fitter.fit_info['nfev']} iterations")
        uncert = np.sqrt(np.diag(cov_mat))

    f_cent = np.zeros(f_guess.shape)
    f_std = np.zeros(f_guess.shape)
    f_cent_e = np.zeros(f_guess.shape)
    f_std_e = np.zeros(f_guess.shape)

    # Make gaussians iterable in case there is only one component.
    try:
        iter(gaussians)
    except TypeError:
        gaussians = (gaussians,)
    # Fit each subburst in frequency.
    for i, gauss in enumerate(gaussians):
        # Use only the inner 2 sigma in time.
        sb_region = (times > gauss.mean-2*gauss.stddev) & (times < gauss.mean+2*gauss.stddev)
        sb_spectrum = waterfall[:, sb_region].mean(axis=1)

        # Fit a Gaussian in frequency.
        weights = (~sb_spectrum.mask).astype(np.int)
        freq_gauss, freq_fitter = fit_multiple_1d_gaussians(freqs, sb_spectrum.data, amp_guess[i:i+1],
                                                  f_guess[i:i+1], f_std_g[i:i+1], weights=weights)

        # Save the fitted parameters.
        f_cent[i], f_std[i] = freq_gauss.mean.value, freq_gauss.stddev.value
        try:
            errors = np.sqrt(np.diag(freq_fitter.fit_info['param_cov']))
            _, f_cent_e[i], f_std_e[i] = errors
        except ValueError:
            print(f"The frequency fit did not work. {freq_fitter.fit_info['message']}")
            f_cent_e[i], f_std_e[i] = np.nan, np.nan

    if make_plot:
        plot_fits(waterfall, times, freqs, time_series, sb_spectrum, gaussians, freq_gauss, offpulse)

    amps, t_cent, t_std = gauss_params[:, 0], gauss_params[:, 1], gauss_params[:, 2]
    amps_e, t_cent_e, t_std_e = uncert[0::3], uncert[1::3], uncert[2::3]
    return amps, amps_e, t_cent, t_cent_e, t_std, t_std_e, f_cent, f_cent_e, f_std, f_std_e


def fit_observation(basename, orig_hdf5_file=None, propery_hdf5=None, pulse=None, tavg=1, subb=1,
                    make_plot=False):
    warnings.filterwarnings("ignore", message="Polarization is AABBCRCI, averaging AA and BB")
    # hdf5 file
    if not orig_hdf5_file:
        orig_hdf5_file = f'{basename}.hdf5'
    if not propery_hdf5:
        property_hdf5 = f'{basename}_burst_properties.hdf5'

    if not os.path.isfile(orig_hdf5_file) and os.path.isdir(basename):
        os.chdir(os.path.join(basename, 'pulses'))

    pulses = pd.read_hdf(property_hdf5, 'pulses')

    # Get pulses to be processed.
    if pulse is not None:
        pulse_ids = [pulse]
    else:
        pulse_ids = pulses.index.get_level_values(0).unique()
        if ('Gaussian', 'Amp') in pulses.columns:
            not_processed = pulses.loc[(slice(None), 'sb1'), ('Gaussian', 'Amp')].isna()
            pulse_ids = pulse_ids[not_processed]

    for pulse_id in pulse_ids:
        print("1D Gaussian fit of observation %s, pulse ID %s" % (basename, pulse_id))
        base_pulse = pulse_id.split('-')[0]
        global pulse_dir
        pulse_dir = f'{base_pulse}/'
        fits_file = pulse_dir + f'{basename}_{base_pulse}.fits'

        # Get some observation parameters
        fits = psrfits.PsrfitsFile(fits_file)
        tsamp = fits.specinfo.dt * tavg
        freqs = fits.freqs

        # Name mask and offpulse file
        maskfile = pulse_dir + f'{basename}_{base_pulse}_mask.pkl'
        offpulsefile = pulse_dir + f'{basename}_{base_pulse}_offpulse_time.pkl'

        # read in DMs and such
        dm = pulses.loc[(pulse_id, 'sb1'), ('General','DM')]
        t_fit = pulses.loc[(pulse_id, 'sb1'), ('General', 'tfit')]

        amp_guesses = pulses.loc[pulse_id, ('Guesses', 'Amp')]
        time_guesses = pulses.loc[pulse_id, ('Guesses', 't_cent')] / tavg
        t_std_guess = pulses.loc[pulse_id, ('Guesses', 't_std')] / 1e3  # ms to s
        freq_peak_guess = pulses.loc[pulse_id, ('Guesses', 'f_cent')]
        f_std_guess = pulses.loc[pulse_id, ('Guesses', 'f_std')]

        # Load the data.
        waterfall, t_ref, _ = import_fil_fits.fits_to_np(
            fits_file, dm=dm, maskfile=maskfile, AO=True,
            hdf5=orig_hdf5_file, index=base_pulse, tavg=tavg)
        waterfall = import_fil_fits.bandpass_calibration(waterfall, offpulsefile, tavg=tavg,
                                                         AO=True, smooth_val=None, plot=False)

        fit_results = fit_time_and_freq(waterfall, tsamp, freqs, offpulsefile, amp_guesses, time_guesses, t_std_guess,
                          freq_peak_guess, f_std_guess, t_fit, tavg, make_plot=make_plot)

        # Save all.
        pulses.loc[pulse_id, ('Gaussian', 'Amp')] = fit_results[0]
        pulses.loc[pulse_id, ('Gaussian', 'Amp e')] = fit_results[1]
        pulses.loc[pulse_id, ('Gaussian', 't_cent / s')] = fit_results[2] + t_ref
        pulses.loc[pulse_id, ('Gaussian', 't_cent_e / s')] = fit_results[3]
        pulses.loc[pulse_id, ('Gaussian', 't_std / ms')] = fit_results[4] * 1e3
        pulses.loc[pulse_id, ('Gaussian', 't_std_e / ms')] = fit_results[5] * 1e3
        pulses.loc[pulse_id, ('Gaussian', 'f_cent / MHz')] = fit_results[6]
        pulses.loc[pulse_id, ('Gaussian', 'f_cent_e / MHz')] = fit_results[7]
        pulses.loc[pulse_id, ('Gaussian', 'f_std / MHz')] = fit_results[8]
        pulses.loc[pulse_id, ('Gaussian', 'f_std_e / MHz')] = fit_results[9]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            pulses.to_hdf(property_hdf5, 'pulses')


def plot_fits(waterfall, times, freqs, time_series, sb_spectrum, gaussians, freq_gauss, offpulse):
    rows = 2
    cols = 2
    gs = gridspec.GridSpec(rows, cols, wspace=0., hspace=0.,
                           height_ratios=[0.5,]*(rows-1) + [2,],
                           width_ratios=[5,] + [1,]*(cols-1))
    fig = plt.figure('1D fit', figsize=(16, 10))
    ax_data = plt.subplot(gs[2])
    ax_ts = plt.subplot(gs[0], sharex=ax_data)
    ax_spec = plt.subplot(gs[-1], sharey=ax_data)
    ax_data.imshow(waterfall, origin='upper', aspect='auto', extent=[times.min(), times.max(), freqs[-1], freqs[0]])

    time_series = waterfall.mean(axis=0)
    off_mean = time_series[offpulse].mean()
    off_std = time_series[offpulse].std()
    time_series = (time_series - off_mean) / off_std

    if isinstance(gaussians, tuple):
        gaussians = gaussians[0]
    ax_ts.plot(times, time_series)
    ax_ts.plot(times, gaussians(times))
    ax_spec.plot(sb_spectrum, freqs)
    ax_spec.plot(freq_gauss(freqs), freqs)

    fig.savefig(pulse_dir+'1D_fit.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('basename',
                        help='The base of the observation (e.g. puppi_58564_C0531+33_0020)')
    parser.add_argument('-p', '--pulse', type=str, help="Give a pulse id to process only this pulse.")
    parser.add_argument('-t', '--tavg', dest='tavg', type=int, default=1,
                      help="If -t option is used, time averaging is applied using the factor "
                           "given after -t.")
    parser.add_argument('-s', '--subb', type=int, default=1,
                      help="If -s option is used, subbanding is applied using the factor "
                           "given after -s.")
    parser.add_argument('-m', '--make_plot', type=bool, default=False,
                      help="Make a waterfaller plot with the fitted functions.")

    args = parser.parse_args()

    fit_observation(**vars(args))
# =============================================================================
#     basename='puppi_58432_C0531+33_0035'
#     fit_observation(basename, pulse='51', tavg=8)
# =============================================================================
