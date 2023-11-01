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
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

from presto import psrfits

import import_fil_fits


def sum_observation_residuals(basename, orig_hdf5_file=None, propery_hdf5=None, pulse=None, tavg=1,
                              subb=1):
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

    mean_array = np.zeros(fits.nchan)
    chans_added = np.zeros(fits.nchan)

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

        spectrum = waterfall.sum(1)
        # freqs = fits.freqs[~spectrum.mask]

        butt = butter_lowpass_filter(spectrum.compressed(), 1, 60, 2)
        buttrum = spectrum
        buttrum[~spectrum.mask] = spectrum[~spectrum.mask] - butt

        mean_array += buttrum
        chans_added += ~spectrum.mask

        # plt.plot(freqs, spectrum)
        # plt.plot(freqs, butt)

        # # Calculate channel weights
        # new_chan_weights = (~waterfall.mask[:, 0]).astype(np.float)

        # add_data_to_mean(waterfall.filled(0), mean_array, weight_stack=n_averaged,
        #                  new_chan_weights=new_chan_weights, mean_chan_weights=mean_chan_weights)

    np.savez(f'{basename}_sum_of_{n_averaged+1}_residuals', n_averaged=len(pulse_ids), freqs=fits.freqs,
             chans_added=chans_added, mean_array=mean_array)

    # Make a plot.
    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'hspace' : 0})
    axs[0].plot(fits.freqs, chans_added)
    axs[1].plot(fits.freqs, mean_array)
    axs[1].set_xlabel("Frequency / MHz")
    axs[1].set_ylabel("Summed spectrum residuals")
    axs[0].set_ylabel("Number a channel is used")
    fig.savefig(f'{basename}_sum_of_{n_averaged+1}_residuals.png')


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def average_averages(file_names):
    """Average any number of averages that were saved as npz files"""
    # Load the first file and take it as the "base" mean array.
    data = np.load(file_names[0])
    n_averaged, freqs, chans_added, mean_array = [data[key] for key in data.files]
    for file in file_names[1:]:
        # Load file to add
        data = np.load(file)
        n_av_new, tsamp1, freqs1, new_chan_weights, new_data = [data[key] for key in data.files]

        # Add it
        # add_data_to_mean(new_data, mean_array, weight_new=n_av_new, weight_stack=n_averaged,
        #                  offset=None, new_chan_weights=new_chan_weights,
        #                  mean_chan_weights=chans_added)
        n_averaged += n_av_new

    # Save new file
    np.savez(f'mean_of_means_of_{n_averaged}', n_averaged=n_averaged,
             freqs=freqs, chans_added=chans_added, mean_array=mean_array)
    # fig = plot_waterfall(mean_array, 1e3*np.arange(tsamp*mean_array.shape[1], step=tsamp),
    #                freqs, chans_added,
                   # title=f'Average burst of {n_averaged} bursts over several observations')
    # fig.savefig(f'average_of_{n_averaged}_bursts.png')


def plot_from_file(file_name):
    data = np.load(file_name)
    n_averaged, freqs, chans_added, mean_array = [data[key] for key in data.files]
    title = os.path.basename(file_name)
    title = os.path.splitext(title)[0]

    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'hspace' : 0})
    axs[0].plot(freqs, chans_added)
    axs[1].plot(freqs, mean_array)
    axs[1].set_xlabel("Frequency / MHz")
    axs[1].set_ylabel("Summed spectrum residuals")
    axs[0].set_ylabel("Number a channel is used")
    fig.savefig(f'{title}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('basename',
                        help='The base of the observation (e.g. puppi_58564_C0531+33_0020)')
    parser.add_argument('-p', '--pulse', type=str,
                        help="Give a pulse id to process only this pulse.")

    args = parser.parse_args()

    sum_observation_residuals(**vars(args))
    # basename = 'puppi_58432_C0531+33_0035'
    # average_observation(basename)
