#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:26:15 2021

@author: jjahns
"""
import sys
#sys.path.append('/home/jjahns/Bursts/AO_burst_properties_pipeline/')
import os
import numpy as np
import matplotlib.pyplot as plt
import import_fil_fits
import matplotlib.gridspec as gridspec

import pandas as pd
import warnings
import argparse

from glob import glob
from presto import psrfits
from fit_bursts import ds

def plot_gallery(basename=None, tavg=8, subb=1, width=20, cut_snr=10):
    warnings.filterwarnings("ignore", message="Polarization is AABBCRCI, averaging AA and BB")

    if not basename == None:
        obsis = [basename]
    else:
        obsis = sorted(glob('puppi*'))


    # Determine numbers of plots
    pulses = []
    for basename in obsis:
        prop_file = f'{basename}/pulses/{basename}_burst_properties.hdf5'
        if not os.path.isfile(prop_file):
            print(f"skipping {basename}")
            continue
        pulses.append(pd.read_hdf(prop_file).sort_index())
    pulses = pd.concat(pulses)
    in_band = (pulses[('Drifting Gaussian', 'f_cent / MHz')] > 1100) & (pulses[('Drifting Gaussian', 'f_cent / MHz')] < 1800)
    bright = ((pulses[('Drifting Gaussian', 'Amp')] > cut_snr) & in_band).groupby(level=0).max()
    n_plots = np.count_nonzero((pulses.groupby(level=0).size() > 2) | bright.squeeze())

    # Prepare plot layout
    cols = 5
    rows = int(np.ceil(n_plots / cols))
    fig = plt.figure(figsize=(cols*6.4, rows*4.8))
    plt.tight_layout()
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    plot_nr = 0
    for basename in obsis: # = 'puppi_58432_C0531+33_0035'
        # Get original hdf5 file to locate the bursts.
        in_hdf5_file = f'{basename}/pulses/{basename}.hdf5'
        pulses_orig = pd.read_hdf(in_hdf5_file, 'pulses')
        pulses_orig = pulses_orig.loc[pulses_orig['Pulse'] == 0].sort_values('Sigma', ascending=False)

        prop_file = f'{basename}/pulses/{basename}_burst_properties.hdf5'
        if not os.path.isfile(prop_file):
            print(f"skipping {basename}")
            continue
        pulses = pd.read_hdf(prop_file, 'pulses').sort_values(('Guesses', 't_cent'))

        in_band = (pulses[('Drifting Gaussian', 'f_cent / MHz')] > 1100) & (pulses[('Drifting Gaussian', 'f_cent / MHz')] < 1800)
        bright = ((pulses[('Drifting Gaussian', 'Amp')] > cut_snr) & in_band).groupby(level=0).max()
        pulse_ids = (pulses.groupby(level=0).size() > 2) | bright.squeeze()
        pulse_ids = pulse_ids[pulse_ids].index

        for pulse_id in pulse_ids:
            print(f"Plotting burst {plot_nr+1} out of {n_plots}.")
            base_pulse = pulse_id.split('-')[0]
            pulse_dir = f'{basename}/pulses/{base_pulse}/'
            filename = pulse_dir + f'{basename}_{base_pulse}.fits'

            # Name mask and offpulse file
            maskfile = pulse_dir + f'{basename}_{base_pulse}_mask.pkl'
            offpulsefile = pulse_dir + f'{basename}_{base_pulse}_offpulse_time.pkl'

            # Skip pulse if fits file does not exist.
            try:
                fits = psrfits.PsrfitsFile(filename)
            except ValueError:
                print(f'File {filename} does not exist.')
                continue

            # Get some observation parameters
            fits = psrfits.PsrfitsFile(filename)
            t_samp = fits.specinfo.dt * tavg
            freqs = fits.frequencies
            mjd = fits.specinfo.mjd

            dm = pulses.loc[(pulse_id, 'sb1'), ('General','DM')]
            time_guesses = pulses.loc[pulse_id, ('Guesses', 't_cent')] / tavg

            waterfall, t_ref, _ = import_fil_fits.fits_to_np(
                filename, dm=dm, maskfile=maskfile, AO=True,
                hdf5=pulses_orig, index=base_pulse, tavg=tavg)
            waterfall = import_fil_fits.bandpass_calibration(waterfall, offpulsefile, tavg=tavg,
                                                             AO=True, smooth_val=None, plot=False)

            # Define the window to be shown.
            begin_samp = int(time_guesses.mean() - width*1e-3 / t_samp)
            end_samp = int(time_guesses.mean() + width*1e-3 / t_samp)
            waterfall = waterfall[:, begin_samp:end_samp]
            t_ref += begin_samp * t_samp

            waterfall = ds(waterfall, factor=subb)
            freqs = ds(freqs, factor=subb)
            plot_burst(waterfall, gs[plot_nr], t_samp, t_ref, freqs, title=f"MJD {mjd}, Burst {pulse_id}")
            plot_nr += 1

    fig.savefig('burst_gallery.png')


def plot_burst(waterfall, gs, t_samp, t_ref, freqs, title=''):
    profile = np.sum(waterfall, axis=0)
    top_freq = 1750
    bottom_freq = 1150
    band = (freqs < top_freq) & (freqs > bottom_freq)
    waterfall = waterfall[band]
    freqs = freqs[band]

    # Plot the whole thing
    subgs = gs.subgridspec(2, 1, wspace=0., hspace=0.,
                                    height_ratios=[0.5, 2])

    ax_data = plt.subplot(subgs[1])  # dynamic spectrum
    ax_ts = plt.subplot(subgs[0], sharex=ax_data)  # time series

    ax_ts.set_title(title)

    ts_plot, = ax_ts.plot(t_ref + t_samp*np.arange(profile.shape[0]), profile, 'k-', alpha=1.0, )
    y_range = profile.max() - profile.min()
    ax_ts.set_ylim(profile.min() - y_range * 0.05, profile.max() + y_range * 0.05)
    plt.setp(ax_ts.get_xticklabels(), visible=False)
    plt.setp(ax_ts.get_xticklines(), visible=False)

    ax_data.imshow(waterfall.filled(0), vmin=-3, vmax=waterfall.max(),
                               interpolation='none', origin='upper', aspect='auto', cmap='inferno',
                               extent=[t_ref, t_ref + t_samp*profile.shape[0], freqs[-1], freqs[0]],
                               )

    #ax_data.set_xticklabels(ax_data.get_xticks(), rotation = 15)
    ax_data.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax_data.ticklabel_format(useOffset=False)
    ax_data.set_xlabel('time / s')
    ax_data.set_ylabel('f / MHz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-b','--basename', type=str, default=None,
                        help='The base of the observation (e.g. puppi_58564_C0531+33_0020)')
    parser.add_argument('-w', '--width', type=float, default=20., help="Width to plot in ms.")
    parser.add_argument('-t', '--tavg', type=int, default=8,
                        help="If -t option is used, time averaging is applied using the factor "
                             "given after -t.")
    parser.add_argument('-s', '--subb', type=int, default=1,
                        help="If -s option is used, subbanding is applied using the factor "
                             "given after -s.")
    parser.add_argument('-c', '--cut_snr', type=int, default=1,
                        help="The lower limit for peak SNR.")

    args = parser.parse_args()
    print(args)
    plot_gallery(**vars(args))

