"""
RFI_zapper adapted only for plotting
"""
import sys
# sys.path.insert(1,'~/FRB_filterbank_tools')
import numpy as np
import matplotlib.pyplot as plt
import import_fil_fits
#from pylab import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import pickle
import os
#import re
import optparse
#import psrfits_pipe as psrfits
from presto import psrfits
import pandas as pd
import warnings

from scipy.stats import normaltest
from fit_bursts import ds
from matplotlib.widgets import Slider

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def one_sided_std(x):
    """Calculate the rms from the median only from values below."""
    x = np.sort(x)[:x.shape[0]//2]
    return np.sqrt(np.mean((x[:-1] - x[-1])**2))


def find_upper_limit(stats, threshold=5.):
    """Finds upper limit for a statistic to be "normal".
    stats : ndarray with some statistic e.g. the one returned by normaltest
    """
    median = np.median(stats)
    rms = one_sided_std(stats)
    return median + threshold*rms


class identify_bursts(object):
    def __init__(self, arr, tsamp, favg, peak_bin, freqs):
        self.peak_times = []
        self.peak_freqs = []
        self.peak_amps = []
        self.peak_bin = peak_bin
        self.tsamp = tsamp
        self.freqs = freqs
        self.tavg = 1
        self.favg = favg
        self.arr = arr
        self.vmin_fac = 1.
        #arr = ds(arr, factor=favg, axis=0)
        profile = np.sum(arr, axis=0)
        spectrum = np.sum(arr, axis=1)

        peak_time = peak_bin*tsamp*1000

        # Plot the whole thing
        rows = 2
        cols = 2
        gs = gridspec.GridSpec(rows, cols, wspace=0., hspace=0.,
                               height_ratios=[0.5,]*(rows-1) + [2,],
                               width_ratios=[5,] + [1,]*(cols-1))

        fig = plt.figure('Identify bursts', figsize=(16, 10))
        #plt.title()
        self.ax_data = plt.subplot(gs[2])  # dynamic spectrum
        self.ax_ts = plt.subplot(gs[0], sharex=self.ax_data)  # time series
        self.ax_spec = plt.subplot(gs[-1], sharey=self.ax_data)  # spectrum
        self.canvas = self.ax_data.figure.canvas

        self.ts_plot, = self.ax_ts.plot(tsamp*1000*np.arange(profile.shape[0]), profile, 'k-', alpha=1.0)
        y_range = profile.max() - profile.min()
        self.ax_ts.set_ylim(profile.min() - y_range * 0.1, profile.max() + y_range * 0.1)
        self.ax_ts.axvline(peak_time - 5, color='r', linestyle='--')
        self.ax_ts.axvline(peak_time + 5, color='r', linestyle='--')

        self.spec_plot, = self.ax_spec.step(spectrum, freqs, 'k-')
        y_range = spectrum.max() - spectrum.min()
        self.ax_spec.set_xlim(spectrum.min() - y_range * 0.1, spectrum.max() + y_range * 0.1)

        self.data_plot = self.ax_data.imshow(arr.filled(0), vmin=arr.min(), vmax=arr.max(),
                                             interpolation='none', origin='upper', aspect='auto',
                                             extent=[0, tsamp*1000*profile.shape[0], freqs[-1], freqs[0]],
                                             )

        self.ax_data.set_xlabel('time / ms')
        plt.colorbar(self.data_plot)
        #fig.add_subplot(self.ax_data)
        #fig.add_subplot(self.ax_ts)
        #fig.add_subplot(self.ax_spec)

        #plt.tight_layout()

        #self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.keyPress = self.canvas.mpl_connect('key_press_event', self.onKeyPress)

        slope = .00003
        off = 0.
        def nu_qube_term(nu, b, e=3):
            return b*70000/(nu/1000)**e
        self.ax_data.plot([peak_time]*freqs.shape[0], freqs)
        line, = self.ax_data.plot(off + peak_time + nu_qube_term(freqs, slope)-nu_qube_term(freqs[0], slope), freqs)

        axoff = plt.axes([0.01, 0.01, 0.3, 0.02])
        axslope = plt.axes([0.6, 0.01, 0.3, 0.02])
        axexp = plt.axes([0.6, 0.03, 0.3, 0.02])
        off_slider = Slider(axoff, 'Offset', peak_time - 10, peak_time + 10, valinit=peak_time)
        slope_slider = Slider(axslope, 'Slope', 0, 2*slope, valinit=slope)
        exp_slider = Slider(axexp, 'Exponent', 0, 6, valinit=3)

        def sliders_on_changed(val):
            off = off_slider.val
            slope = slope_slider.val
            expo = exp_slider.val
            line.set_xdata(off + nu_qube_term(freqs, slope, e=expo)-nu_qube_term(freqs[0], slope, e=expo))
            fig.canvas.draw_idle()
        off_slider.on_changed(sliders_on_changed)
        slope_slider.on_changed(sliders_on_changed)
        exp_slider.on_changed(sliders_on_changed)

        plt.show()
        #indices = np.array(self.peak_times).argsort()
        #self.peak_times = list(np.array(self.peak_times)[indices])  # I guess this is too late
        #self.peak_amps = list(np.array(self.peak_amps)[indices])

    def onpress(self, event):
        tb = plt.get_current_fig_manager().toolbar
        if tb.mode == '':
            y1 = int(round(event.ydata))
            x1 = event.xdata
            self.peak_times.append(x1)
            self.peak_freqs.append(y1)
            #maxval = np.max(self.arr[:, int(x1)])
            meanval = np.mean(self.arr[y1-5:y1+6, int(x1)-5:int(x1)+6])
            self.peak_amps.append(meanval)  # correct for downsampling
            self.ax_data.scatter(x1, y1, lw=1, color='r', marker='x', s=100,
                                 zorder=10)
            print(x1, y1, meanval)
            plt.draw()
    def onKeyPress(self, event):
        if event.key == 'i':
            self.tavg *= 2
        elif event.key == 'y':
            self.favg *= 2
        elif event.key == 'u':
            if self.tavg > 1:
                self.tavg //= 2
        elif event.key == 't':
            if self.favg > 1:
                self.favg //= 2
        elif event.key == 'e':
            self.vmin_fac -= .1
            if self.vmin_fac < .1:
                self.vmin_fac = 1.

        arr = ds(ds(self.arr, factor=self.favg), factor=self.tavg, axis=1)
        if event.key in 'iyut':
            profile = np.sum(arr, axis=0)
            spectrum = np.sum(arr, axis=1)

            # Replot.
            self.ts_plot.set_data(np.linspace(0, profile.shape[0]*self.tavg*self.tsamp*1000,
                                              num=profile.shape[0]),
                                  profile)
            y_range = profile.max() - profile.min()
            self.ax_ts.set_ylim(profile.min() - y_range * 0.1, profile.max() + y_range * 0.1)

            self.spec_plot.set_data(spectrum, ds(self.freqs,  factor=self.favg))
            y_range = spectrum.max() - spectrum.min()
            self.ax_spec.set_xlim(spectrum.min() - y_range * 0.1, spectrum.max() + y_range * 0.1)

            self.data_plot.set_data(arr.filled(0))

        self.data_plot.set_clim(vmin=self.vmin_fac*arr.min(), vmax=arr.max())
        plt.draw()


if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Polarization is AABBCRCI, averaging AA and BB")

    parser = optparse.OptionParser(usage='%prog [options] infile_basename',
                                   description="Interactive RFI zapper")

    parser.add_option('-f', '--favg', dest='favg', type='int', default=1,
                      help="If -f option is used, frequency averaging is applied after RFI "
                      "exision using the factor given after -f.")
    parser.add_option('-t', '--tavg', dest='tavg', type='int', default=1,
                      help="If -t option is used, time averaging is applied using the factor "
                           "given after -t.")
    parser.add_option('-d', '--dm', dest='dm', type='float', default=None,
                      help="If -d option is used, the DM correction will be applied using -d "
                           "<num> value, else it will use the single pulse search determined "
                           "value.")
    parser.add_option('-m', '--mask', dest='mask', type='string', default=None,
                      help="-m <filename of mask file>. Use this to set an initial mask (if "
                           "there are channels that are always contaminated for example).")
    parser.add_option('-p', '--pulse', type='str', default=None,
                      help="Give a pulse id to process only this pulse.")
    parser.add_option('-a', '--autoflag', action='store_false', default=True,
                      help="Do not autoflag channels.")
    parser.add_option('-b', '--burst_position', action='store_true', default=False,
                      help="Only change the burst positions.")

    options, args = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    elif len(args) != 1:
        sys.stderr.write("Only one input file must be provided!\n")
    else:
        options.infile = args[-1]

    basename = options.infile
    #pulses_txt = 'pulse_nos.txt'
    in_hdf5_file = f'{basename}.hdf5'

    pulses_hdf5 = pd.read_hdf(in_hdf5_file, 'pulses')
    pulses_hdf5 = pulses_hdf5.loc[pulses_hdf5['Pulse'] == 0].sort_values('Sigma', ascending=False)

    # Find pulses in this dataset
    if options.pulse is not None:
        pulses = [int(options.pulse)]
    else:
        pulses = pulses_hdf5.index.to_list()
        # Exclude alredy processed pulses
        pulses = [pulse_id for pulse_id in pulses
                  if not os.path.isfile(f'{pulse_id}/{basename}_{pulse_id}_mask.pkl')]

    smooth = None  # smoothing window
    tavg = options.tavg
    favg = options.favg
    auto_flag = options.autoflag

    prop_file = f'{basename}_burst_properties.hdf5'
    if os.path.isfile(prop_file):
        prop_df = pd.read_hdf(prop_file, 'pulses')
    else:
        index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['pulse_id', 'subburst'])
        cols = pd.MultiIndex.from_tuples([('General','DM'),
                                          ('Guesses', 'Amp'),
                                          ('Guesses', 't_cent'),
                                          ('Guesses', 'f_cent')])
        prop_df = pd.DataFrame(index=index, columns=cols, dtype=np.float)
        prop_df.insert(1, ('General', 'Dropouts'), False)

    #print(prop_df)
    in_hdf5_file = f'../{in_hdf5_file}'

    for pulse_id in pulses:
        ind1 = []   # for the burst name indices
        ind2 = []   # for the sub burst name indices

        print("RFI zapping of observation %s, pulse ID %s" % (basename, pulse_id))

        os.chdir(str(pulse_id))
        filename = f'{basename}_{pulse_id}.fits'

        if options.dm is None:
            # use the DM from detection to de-disperse in this initial stage
            dm = pulses_hdf5.loc[pulse_id, 'DM']
        else:
            dm = options.dm

        # Skip pulse if fits file does not exist.
        try:
            fits = psrfits.PsrfitsFile(filename)
        except ValueError:
            print(f'File {filename} does not exist.')
            os.chdir('..')
            continue

        if options.mask is not None:
            initial_mask = options.mask
        elif os.path.isfile(f'{basename}_{pulse_id}_mask.pkl'):
            initial_mask = f'{basename}_{pulse_id}_mask.pkl'
            auto_flag = False
        elif os.path.isfile('../mask.pkl'):
            initial_mask = '../mask.pkl'
        else:
            initial_mask = None

        offpulsefile = '%s_%s_offpulse_time.pkl' % (basename, pulse_id)

        #total_N=fits.specinfo.N / tavg
        t_samp = fits.specinfo.dt * tavg
        freqs = fits.frequencies
        tot_freq = fits.specinfo.num_channels
        #total_N = int(100e-3 / (t_samp * tavg))

        # Load data
        data, _, peak_bin = import_fil_fits.fits_to_np(filename, dm=dm, maskfile=initial_mask,
                                                           AO=True, hdf5=in_hdf5_file,
                                                           index=pulse_id, tavg=tavg)

        print("Please click where each sub-burst peak of the main burst is.")
        print("Use the keys i and u to downsample and y, t to subband and e to lower vmin.")
        data = import_fil_fits.bandpass_calibration(data, offpulsefile, tavg=tavg, AO=True)
        burst_id = identify_bursts(data, t_samp, favg, peak_bin, freqs)
        plt.close('all')

