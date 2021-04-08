#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:20:48 2021

@author: jjahns
"""
import sys
sys.path.append('/home/jjahns/Bursts/AO_burst_properties_pipeline/')
import os
import argparse
import numpy as np
import pandas as pd
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import import_fil_fits

from presto import psrfits


def average_observation(basename, orig_hdf5_file=None, propery_hdf5=None, pulse=None, tavg=1, subb=1,
                      tstack_window=30.):
    warnings.filterwarnings("ignore", message="Polarization is AABBCRCI, averaging AA and BB")
    print(f"Averaging observation {basename}.")
    # hdf5 file
    if not orig_hdf5_file:
        orig_hdf5_file = f'{basename}.hdf5'
    if not propery_hdf5:
        property_hdf5 =  f'{basename}_burst_properties.hdf5'

    if not os.path.isfile(orig_hdf5_file) and os.path.isdir(basename):
        os.chdir(os.path.join(basename, 'pulses'))

    pulses = pd.read_hdf(property_hdf5, 'pulses')

    # Get pulses to be processed.
    if pulse is not None:
        pulse_ids = [pulse]
    else:
        pulse_ids = pulses.index.get_level_values(0).unique()

    # Initialize the averaged array.
    base_pulse = pulse_ids[0].split('-')[0]
    pulse_dir = f'{base_pulse}/'
    fits_file = pulse_dir + f'{basename}_{base_pulse}.fits'
    fits = psrfits.PsrfitsFile(fits_file)
    tsamp = fits.specinfo.dt * tavg
    n_samps = int(tstack_window/1e3/tsamp)

    mean_array = np.zeros((fits.nchan, n_samps))
    mean_chan_weights = np.zeros(fits.nchan)

    for n_averaged, pulse_id in enumerate(pulse_ids):
        print(f"Adding pulse {pulse_id}")
        base_pulse = pulse_id.split('-')[0]
        pulse_dir = f'{base_pulse}/'
        fits_file = pulse_dir + f'{basename}_{base_pulse}.fits'

        # Name mask and offpulse file
        maskfile = pulse_dir + f'{basename}_{base_pulse}_mask.pkl'
        offpulsefile = pulse_dir + f'{basename}_{base_pulse}_offpulse_time.pkl'

        # Get some observation parameters
        fits = psrfits.PsrfitsFile(fits_file)
        tsamp = fits.specinfo.dt * tavg

        # read in DMs and such
        dm = pulses.loc[(pulse_id, 'sb1'), ('General','DM')]
        time_guesses = pulses.loc[pulse_id, ('Guesses', 't_cent')] / tavg
        t_fit = pulses.loc[(pulse_id, 'sb1'), ('General', 'tfit')]

        # Load the data.
        waterfall, t_ref, _ = import_fil_fits.fits_to_np(
            fits_file, dm=dm, maskfile=maskfile, AO=True,
            hdf5=orig_hdf5_file, index=base_pulse, tavg=tavg)

        waterfall = import_fil_fits.bandpass_calibration(waterfall, offpulsefile, tavg=tavg,
                                                         AO=True, smooth_val=None, plot=False)

        # Get the widow of the burst.
        window_start = int(time_guesses.min() - t_fit*1e-3/tsamp)
        widow_end = int(time_guesses.max() + t_fit*1e-3/tsamp)
        waterfall = waterfall[:, window_start:widow_end]

        # Calculate channel weights
        new_chan_weights = (~waterfall.mask[:, 0]).astype(np.float)

        add_data_to_mean(waterfall.filled(0), mean_array, weight_stack=n_averaged,
                         new_chan_weights=new_chan_weights, mean_chan_weights=mean_chan_weights)

    np.savez(f'{basename}_mean_of_{n_averaged+1}', n_averaged=n_averaged+1, tsamp=tsamp,
             freqs=fits.freqs, chan_weights=mean_chan_weights, mean_array=mean_array)

    # Make a plot.
    fig = plot_waterfall(mean_array, 1e3*np.arange(tsamp*mean_array.shape[1], step=tsamp),
                         fits.freqs, mean_chan_weights)
    fig.savefig(f'{basename}_average_profile.png')


def add_data_to_mean(new_data, mean_array, weight_new=1, weight_stack=1, offset=None,
                     new_chan_weights=None, mean_chan_weights=None):
    """Add a new array to an array of an average burst spectrum.

    Parameters
    ----------
    new_data : array like
        Data to be added to the mean_array.
    mean_array : array like
        Array holding the mean of data previously taken.
    weight_new : int, optional
        Number of bursts that were allready averaged to get new_data.
    weight_stack : int, optional
        Number of bursts that were already averaged to get the mean array.
    offset : int, optional
        Offset in samples from the start where the new_data should be added in mean_array.

    Returns
    -------
    Nothing, but changes mean_array and mean_chan_weights if given.

    """
    # Test if frequency dimension is equal
    if not new_data.shape[0] == mean_array.shape[0]:
        raise ValueError("The number of frequency channels has to be the same.")

    # Calculate window the new data will be written in
    len_new = new_data.shape[1]
    len_stack = mean_array.shape[1]

    if offset is None:
        offset = (len_stack - len_new) // 2
    start = offset
    end = len_new + offset

    # Check if the new data fits inside the stack array.
    if start < 0:
        warnings.warn(f"Data to be added is {start} samples too long or offset negative, the "
                      "data will be cut at the start.")
        new_data = new_data[:, -start:]
        start = 0

    if end > len_stack:
        warnings.warn(f"Data to be added is {(end-len_stack)} samples too long or offset too "
                      "big, the data will be cut.")
        new_data = new_data[:, :-(end-len_stack)]

    # Get it weighted by the new number of stacked bursts and sum it up.
    mean_array *= weight_stack / (weight_stack+weight_new)
    new_data = new_data * weight_new / (weight_stack+weight_new)

    mean_array[:, start:end] += new_data

    # Calculate the channel weights in the same way.
    if new_chan_weights is not None and mean_chan_weights is not None:
        mean_chan_weights *= weight_stack / (weight_stack+weight_new)
        new_chan_weights = new_chan_weights * weight_new / (weight_stack+weight_new)

        mean_chan_weights[:] += new_chan_weights


def plot_waterfall(waterfall, times, freqs, weights, title=''):
    """Make a simple waterfaller plot. Mostly copied from burst_gallery.py"""
    rows = 2
    cols = 3
    gs = gridspec.GridSpec(rows, cols, wspace=0., hspace=0.,
                           height_ratios=[0.5,]*(rows-1) + [2,],
                           width_ratios=[5,] + [1,]*(cols-1))
    fig = plt.figure('Average burst profile', figsize=(16, 10))
    fig.suptitle(title, fontsize=16)
    ax_data = plt.subplot(gs[3])
    ax_ts = plt.subplot(gs[0], sharex=ax_data)
    ax_spec = plt.subplot(gs[-2], sharey=ax_data)
    ax_weight = plt.subplot(gs[-1], sharey=ax_data)

    top_freq = 1750
    bottom_freq = 1150
    band = (freqs < top_freq) & (freqs > bottom_freq)
    waterfall = waterfall[band]
    freqs = freqs[band]
    weights = weights[band]

    # Time series
    time_series = waterfall.mean(axis=0)
    ts_plot, = ax_ts.plot(times, time_series, 'k-')
    y_range = time_series.max() - time_series.min()
    ax_ts.set_ylim(time_series.min() - y_range * 0.05, time_series.max() + y_range * 0.05)
    plt.setp(ax_ts.get_xticklabels(), visible=False)
    plt.setp(ax_ts.get_xticklines(), visible=False)

    # Waterfall
    ax_data.imshow(waterfall, vmin=waterfall.min(), vmax=waterfall.max(),
                   interpolation='none', origin='upper', aspect='auto', cmap='inferno',
                   extent=[times.min(), times.max(), freqs[-1], freqs[0]],
                   )

    # Spectrum
    ax_spec.plot(waterfall.mean(axis=1), freqs, 'k-')
    #ax_spec.set_xlim([0, None])

    # Channel weights
    ax_weight.step(weights, freqs, 'k-', linewidth=0.5)
    ax_weight.set_xlim([-.05, 1.05])
    ax_weight.yaxis.set_label_position('right')
    ax_weight.yaxis.tick_right()

    #ax_data.ticklabel_format(useOffset=False)
    ax_data.set_xlabel('time / ms')
    ax_data.set_ylabel('f / MHz')

    return fig


def average_averages(file_names):
    """Average any number of averages that were saved as npz files"""
    # Load the first file and take it as the "base" mean array.
    data = np.load(file_names[0])
    n_averaged, tsamp, freqs, chan_weights, mean_array = [data[key] for key in data.files]
    for file in file_names[1:]:
        # Load file to add
        data = np.load(file)
        n_av_new, tsamp1, freqs1, new_chan_weights, new_data = [data[key] for key in data.files]

        # Check if compatible
        if tsamp1 != tsamp or np.any(freqs1 != freqs):
            warnings.warn("The files you are trying to average are not compatible in time or frequency")

        # Add it
        add_data_to_mean(new_data, mean_array, weight_new=n_av_new, weight_stack=n_averaged,
                         offset=None, new_chan_weights=new_chan_weights,
                         mean_chan_weights=chan_weights)
        n_averaged += n_av_new

    # Save new file
    np.savez(f'mean_of_means_of_{n_averaged}', n_averaged=n_averaged, tsamp=tsamp,
             freqs=freqs, chan_weights=chan_weights, mean_array=mean_array)
    plot_waterfall(mean_array, 1e3*np.arange(tsamp*mean_array.shape[1], step=tsamp),
                   freqs, chan_weights,
                   title=f'Average burst of {n_averaged} bursts over several observations')


def plot_from_file(file_name, titel=''):
    data = np.load(file_name)
    n_averaged, tsamp, freqs, chan_weights, mean_array = [data[key] for key in data.files]
    plot_waterfall(mean_array, 1e3*np.arange(tsamp*mean_array.shape[1], step=tsamp),
                   freqs, chan_weights, title=titel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('basename',
                        help='The base of the observation (e.g. puppi_58564_C0531+33_0020)')
    parser.add_argument('-p', '--pulse', type=str, help="Give a pulse id to process only this pulse.")
    parser.add_argument('-t', '--tavg', dest='tavg', type=int, default=1,
                      help="If -t option is used, time averaging is applied using the factor "
                           "given after -t.")

    args = parser.parse_args()

    average_observation(**vars(args))
# =============================================================================
#     basename='puppi_58432_C0531+33_0035'
#     average_observation(basename, tavg=8)
# =============================================================================
