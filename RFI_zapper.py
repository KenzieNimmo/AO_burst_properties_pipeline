"""
FOR ARECIBO FRB121102 ANALYSIS

Interactive RFI masker using the fits file, and outputs the masked array that can be read
into other python scripts. Also does the bandpass correction using import_fil_fits.
Kenzie Nimmo 2020
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
    def __init__(self, arr, tsamp, favg, peak_bin):
        self.peak_times = []
        self.peak_freqs = []
        self.peak_amps = []
        self.peak_bin = peak_bin
        self.tsamp = tsamp
        self.tavg = 1
        self.favg = favg
        self.arr = arr
        self.vmin_fac = 1.
        #arr = ds(arr, factor=favg, axis=0)
        profile = np.sum(arr, axis=0)
        spectrum = np.sum(arr, axis=1)

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

        self.ts_plot, = self.ax_ts.plot(profile, 'k-', alpha=1.0)
        y_range = profile.max() - profile.min()
        self.ax_ts.set_ylim(profile.min() - y_range * 0.1, profile.max() + y_range * 0.1)
        self.ax_ts.axvline(peak_bin - (5e-3 / tsamp), color='r', linestyle='--')
        self.ax_ts.axvline(peak_bin + (5e-3 / tsamp), color='r', linestyle='--')

        self.spec_plot, = self.ax_spec.step(spectrum, np.arange(spectrum.shape[0]), 'k-')
        y_range = spectrum.max() - spectrum.min()
        self.ax_spec.set_xlim(spectrum.min() - y_range * 0.1, spectrum.max() + y_range * 0.1)

        self.data_plot = self.ax_data.imshow(arr.filled(0), vmin=arr.min(), vmax=arr.max(),
                                             origin='upper', aspect='auto')

        plt.colorbar(self.data_plot)
        #fig.add_subplot(self.ax_data)
        #fig.add_subplot(self.ax_ts)
        #fig.add_subplot(self.ax_spec)

        plt.tight_layout()

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.keyPress = self.canvas.mpl_connect('key_press_event', self.onKeyPress)

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
            self.ts_plot.set_data(np.arange(profile.shape[0]*self.tavg, step=self.tavg), profile)
            y_range = profile.max() - profile.min()
            self.ax_ts.set_ylim(profile.min() - y_range * 0.1, profile.max() + y_range * 0.1)

            self.spec_plot.set_data(spectrum, np.arange(spectrum.shape[0]*self.favg, step=self.favg))
            y_range = spectrum.max() - spectrum.min()
            self.ax_spec.set_xlim(spectrum.min() - y_range * 0.1, spectrum.max() + y_range * 0.1)

            self.data_plot.set_data(arr.filled(0))

        self.data_plot.set_clim(vmin=self.vmin_fac*arr.min(), vmax=arr.max())
        plt.draw()



class offpulse(object):
    def __init__(self, arr, gs, tsamp, peak_bin, initial_mask=None):
        self.begin_times = []
        self.end_times = []
        self.lines = {}

        ax1 = plt.subplot(gs[3])  # dynamic spectrum
        ax2 = plt.subplot(gs[0], sharex=ax1)  # profile
        ax3 = plt.subplot(gs[-1], sharey=ax1)  # spectrum
        ax4 = plt.subplot(gs[-2], sharey=ax1)
        self.dyn_spec = ax1
        self.spec = ax3
        self.stat = ax4  # To plot channel statistics

        self.axes = ax2  # off pulse only necessary for the profile which is in subplot ax2

        self.canvas = ax2.figure.canvas

        profile = np.mean(arr, axis=0)

        self.ax2plot, = ax2.plot(profile, 'k-', alpha=1.0, zorder=1)
        ax2.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        ax2.tick_params(axis='x', labelbottom='off', top='off')
        y_range = profile.max() - profile.min()
        ax2.set_ylim(profile.min() - y_range * 0.1, profile.max() + y_range * 0.1)
        ax2.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                        labelright=False, bottom=True, top=True, left=True, right=True)
        ax2.axvline(peak_bin - (5e-3 / tsamp), color='r', linestyle='--')
        ax2.axvline(peak_bin + (5e-3 / tsamp), color='r', linestyle='--')
        fig.add_subplot(ax2)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.crel = self.canvas.mpl_connect('button_release_event', self.onrel)
        self.keyPress = self.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.keyRelease = self.canvas.mpl_connect('key_release_event', self.onKeyRelease)

        self.data = self.ax2plot.get_data()
        self.profile = self.data[1]
        self.x = False

    def clear_line(self, x):
        self.lines.pop(x).remove()

    def onKeyPress(self, event):
        if event.key  in ('x', 'u'):
            self.x = True
        if event.key == 'y':
            if self.lines['burst']:
                self.clear_line('burst')
                plt.draw()

    def onKeyRelease(self, event):
        if event.key  in ('x', 'u'):
            self.x = False

    def onpress(self, event):
        # if self.ctrlKey == True and self.shiftKey == True:
        if self.x:
            tb = plt.get_current_fig_manager().toolbar
            if tb.mode == '':
                x1 = event.xdata
                index1 = np.int(x1)
                self.begin_times.append(index1)
        if not self.x:
            return

    def onrel(self, event):
        # if self.ctrlKey == True and self.shiftKey == True:
        if self.x:
            tb = plt.get_current_fig_manager().toolbar
            if tb.mode == '':
                x2 = event.xdata
                index2 = np.int(x2)
                self.end_times.append(index2)
                if self.begin_times[-1] < index2:
                    self.lines['burst'] = self.axes.axvspan(
                        self.begin_times[-1], index2, color='#FF00FF', alpha=0.5, zorder=0.8)
                else:
                    self.lines['burst'] = self.axes.axvspan(
                        index2, self.begin_times[-1], color='#FF00FF', alpha=0.5, zorder=0.8)
                plt.draw()
            if not self.x:
                return


class RFI(object):
    def __init__(self, arr, spectrum, gs, prof, dyn_spec, spec, stat, ithres, ax2, tavg,
                 initial_mask=None, auto_flag=True):
        self.begin_chan = []
        self.mask_chan = []
        if initial_mask is not None:
            initial = pickle.load(open(initial_mask, 'rb'))
            self.mask_chan = [int(x) for x in initial]
        self.axes = dyn_spec  # off pulse only necessary for the profile which is in subplot ax2
        self.canvas = dyn_spec.figure.canvas
        self.ithres = ithres
        self.total_N = arr.shape[1]

# =============================================================================
#         if favg > 1:
#             print("Downsampling is not tested and will most likely fail")
#             nchan_tot = arr.shape[0]
#             favg = float(favg)
#             if (nchan_tot / favg) - int(nchan_tot / favg) != 0:
#                 print(f"The total number of channels is {nchan_tot}, please choose an fscrunch "
#                       + "value that divides the total number of channels."
#                       )
#                 sys.exit()
#             else:
#                 newnchan = nchan_tot / favg
#                 arr = np.array(np.row_stack([np.mean(subint, axis=0)
#                                              for subint in np.vsplit(arr, newnchan)]))
# =============================================================================

        #spectrum = np.mean(arr, axis=1)
        self.spectrum = spectrum
        self.nchans = len(spectrum)
        self.freqbins = np.arange(0, arr.shape[0], 1)
        threshold = np.amax(arr) - (np.abs(np.amax(arr)-np.amin(arr)) * self.ithres)

        self.cmap = mpl.cm.viridis #inferno
        self.ax1 = dyn_spec
        self.ax3 = spec
        self.ax2 = ax2
        self.ax4 = stat
        self.ax2plot = prof
        self.ax1plot = self.ax1.imshow(arr,
                                       aspect='auto',
                                       vmin=np.amin(arr),
                                       vmax=threshold,
                                       cmap=self.cmap,
                                       origin='upper',
                                       #interpolation='nearest',
                                       picker=True)
        self.cmap.set_over(color='pink')
        #self.cmap.set_bad(color='midnightblue')
        #self.ax1.set_xlim((peak_bin - (50e-3 / (tsamp * tavg))),
        #                  (peak_bin + (50e-3 / (tsamp * tavg))))

        self.ax3plot, = self.ax3.plot(spectrum, self.freqbins, 'k-', zorder=2)
        self.ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
        self.ax3.tick_params(axis='y', labelleft='off')
        self.ax3.set_ylim(self.freqbins[-1], self.freqbins[0])
        #x_range = spectrum.max() - spectrum.min()
        self.ax3.set_xlim(spectrum.min(), spectrum.max())  #-x_range / 4., x_range * 6. / 5.)

        # Plot channel statistics
        self.normal_deviation = normaltest(arr.data, axis=1)[0]
        self.ax4plot, = self.ax4.plot(self.normal_deviation, self.freqbins, 'k-', zorder=2)
        self.ax4.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
        self.ax4.tick_params(axis='y', labelleft='off')
        self.ax4.set_ylim(self.freqbins[-1], self.freqbins[0])
        self.ax4.set_xlim(np.nanmin(self.normal_deviation), np.nanmax(self.normal_deviation))

        # Automatically find some bad channels
        mask = np.zeros(self.nchans, dtype=bool)
        mask[self.mask_chan] = True
        self.normal_deviation = np.ma.masked_where(mask, self.normal_deviation)
        self.arr = arr

        if auto_flag:
            self.do_auto_flag()

        fig.add_subplot(self.ax1)
        fig.add_subplot(self.ax4)
        fig.add_subplot(self.ax3)

        plt.tight_layout()

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.crel = self.canvas.mpl_connect('button_release_event', self.onrel)
        self.keyPress = self.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.keyRelease = self.canvas.mpl_connect('key_release_event', self.onKeyRelease)
        self.x = False
        self.r = False

    def do_auto_flag(self):
        mask = self.normal_deviation.mask.copy()
        norm_thres = find_upper_limit(self.normal_deviation.compressed(), threshold=5.)
        bad_chans = self.normal_deviation > norm_thres
        self.arr.mask[bad_chans] = True
        self.normal_deviation.mask[bad_chans] = True
        self.mask_chan.extend((~mask & bad_chans).nonzero()[0])

        self.ax4.axvline(norm_thres, color='r')

    def onKeyPress(self, event):
        if event.key in ('x', 'u'):
            self.x = True
        if event.key == 'r':
            self.r = True
            arr = self.ax1plot.get_array()
            self.ithres -= 0.005
            threshold = np.amax(arr) - (np.abs(np.amax(arr) - np.amin(arr)) * self.ithres)
            self.ax1plot.set_clim(vmin=np.amin(arr), vmax=threshold)
            self.cmap.set_over(color='pink')
            plt.draw()
        if event.key == 'a':
            self.do_auto_flag()

    def onKeyRelease(self, event):
        if event.key in ('x', 'u'):
            self.x = False
        if event.key == 'r':
            self.r = False

    def onpress(self, event):
        # if self.ctrlKey == True and self.shiftKey == True:
        if self.x:
            return
        if not self.x:
            tb = plt.get_current_fig_manager().toolbar
            if tb.mode == '':
                y1 = event.ydata
                index = find_nearest(self.freqbins, y1)
                self.begin_chan.append(index)

        """
        if self.y == True:
            tb = plt.get_current_fig_manager().toolbar
            if tb.mode == '':
                y_range = self.profile.max() - self.profile.min()
                ymin=self.profile.min()-y_range*0.1
                if begin_times[-1] < end_times[-1]:
                    ax.hlines(y=ymin, xmin=begin_times[-1], xmax=end_times[-1], lw=10,
                              color='#FFFFFF',zorder=1.0)
                else:
                    ax.hlines(y=ymin, xmin=end_times[-1], xmax=begin_times[-1], lw=10,
                              color='#FFFFFF',zorder=1.0)
        """

    def onrel(self, event):
        # if self.ctrlKey == True and self.shiftKey == True:
        if self.x:
            return
        if not self.x:
            tb = plt.get_current_fig_manager().toolbar
            if tb.mode == '':
                y2 = event.ydata
                arr = self.arr  # ax1plot.get_array()
                vmin = np.amin(arr)
                index2 = find_nearest(self.freqbins, y2)
                if self.begin_chan[-1] > index2:
                    arr[index2:self.begin_chan[-1] + 1, :] = vmin - 100
                else:
                    arr[self.begin_chan[-1]:index2 + 1, :] = vmin - 100
                mask = arr < vmin - 50
                arr = np.ma.masked_where(mask, arr)
                self.arr = arr
                self.ax1plot.set_data(arr.filled(0))
                profile = np.mean(arr, axis=0)
                self.ax2plot.set_data(np.arange(0, self.total_N, 1), profile)

                self.spectrum = np.ma.masked_where(mask.any(axis=1), self.spectrum)
                self.ax3plot.set_data(self.spectrum, self.freqbins)
                #normal_deviation = normaltest(arr, axis=1)[0]
                self.normal_deviation[mask[:, 0]] = True
                self.ax4plot.set_data(self.normal_deviation, self.freqbins)

                threshold = np.amax(arr) - (np.abs(np.amax(arr) - np.amin(arr)) * self.ithres)
                self.ithres -= 0.005
                self.ax1plot.set_clim(vmin=np.amin(arr), vmax=threshold)

                self.ax3.set_xlim(np.min(self.spectrum), np.max(self.spectrum))
                self.ax4.set_xlim(np.amin(self.normal_deviation), np.amax(self.normal_deviation))
                y_range = profile.max() - profile.min()
                self.ax2.set_ylim(profile.min() - y_range * 0.1, profile.max() + y_range * 0.1)

                self.cmap.set_over(color='pink')
                plt.draw()
                if self.begin_chan[-1] > index2:
                    for i in range(len(np.arange(index2, self.begin_chan[-1] + 1, 1))):
                        self.mask_chan.append(
                            np.arange(index2, self.begin_chan[-1] + 1, 1)[i])
                            #self.nchan - 1 - np.arange(index2, self.begin_chan[-1] + 1, 1)[i])
                else:
                    for i in range(len(np.arange(self.begin_chan[-1], index2 + 1, 1))):
                        self.mask_chan.append(
                            np.arange(self.begin_chan[-1], index2 + 1, 1)[i])
                            #self.nchan - 1 - np.arange(self.begin_chan[-1], index2 + 1, 1)[i])

                self.final_spec = np.mean(arr, axis=1)
                self.final_prof = np.mean(arr, axis=0)


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
        if not options.burst_position:
            rows = 2
            cols = 3
            fig = plt.figure('RFI zapper', figsize=(16, 10))
            gs = gridspec.GridSpec(rows, cols, wspace=0., hspace=0.,
                                   height_ratios=[0.5,]*(rows-1) + [2,],
                                   width_ratios=[5,] + [1,]*(cols-1))
            # Correct the bandpass already, but plot the real spectrum
            spectrum = np.mean(data.data, axis=1)
            data.data[:] -= spectrum[:, np.newaxis]
            nonzero_chans = (data.data != 0.).any(axis=1)  # To avoid deviding by zero
            data.data[nonzero_chans] /= np.std(data.data[nonzero_chans], axis=1)[:, np.newaxis]

            # Make figure and define offpulse region.
            offpulse_prof = offpulse(data, gs, t_samp, peak_bin, initial_mask=initial_mask)
            dyn_spec = offpulse_prof.dyn_spec
            spec = offpulse_prof.spec
            prof = offpulse_prof.ax2plot
            ax2 = offpulse_prof.axes
            stat = offpulse_prof.stat

            # instructions
            print("Click and drag on the dynamic spectrum to identify frequency channels to mask.\n"
                  "Hold x or u and click on the profile and drag to identify regions to be excluded "
                  "from the bandpass calibration.\n"
                  "Click y to remove this selection.\n"
                  "Click r at any time to refresh the plot and lower the threshold for the pink "
                  "points (incase there is too much pink points for example).\n"
                  "Click a to autoflag.")
            ithres = 0.
            RFImask = RFI(data, spectrum, gs, prof, dyn_spec, spec, stat, ithres, ax2, tavg,
                          initial_mask=initial_mask, auto_flag=auto_flag)
            plt.show()

            # Save the maskfile already
            mask_chans = np.array(RFImask.mask_chan)
            maskfile = '%s_%s_mask.pkl' % (basename, pulse_id)
            with open(maskfile, 'wb') as fmask:
                pickle.dump(mask_chans, fmask)
            data.mask[mask_chans] = True

            profile = RFImask.final_prof

            begin_times = np.array(offpulse_prof.begin_times)
            end_times = np.array(offpulse_prof.end_times)
            total_N = len(profile)

            # Save the offpulse as if it was not downsampled
            if tavg != 1:
                total_N *= tavg
                begin_times *= tavg
                end_times *= tavg

            off_pulse = np.ones(total_N, dtype=np.bool)
            if begin_times.size == 0:
                raise ValueError("Warning: you have not defined the off burst region")
            if len(begin_times) != len(end_times):
                raise ValueError("The number of begin and end times doesn't match, make sure to hold "
                                 "x")
            else:
                print(f"Excluding {begin_times.size} windows")
                print(begin_times, end_times)
                for begin_time, end_time in zip(begin_times, end_times):
                    if begin_time < end_time:
                        off_pulse[begin_time:end_time] = False
                    else:
                        off_pulse[begin_time:end_time:-1] = False

            if begin_times.size > 0:
                with open(offpulsefile, 'wb') as foff:
                    pickle.dump(off_pulse, foff)

        print("Please click where each sub-burst peak of the main burst is.")
        print("Use the keys i and u to downsample and y, t to subband and e to lower vmin.")
        data = import_fil_fits.bandpass_calibration(data, offpulsefile, tavg=tavg, AO=True)
        burst_id = identify_bursts(data, t_samp, favg, peak_bin)
        plt.close('all')

        n_subbursts = len(burst_id.peak_times)
        DMs = n_subbursts*[dm]
        for j in range(n_subbursts):
            ind1.append(str(pulse_id))
            ind2.append('sb' + str(j + 1))

        peak_times = burst_id.peak_times
        peak_freqs = burst_id.peak_freqs
        amps = burst_id.peak_amps

        # Are there other bursts in this file?
        answer = input("Are there other (separate) bursts in this file? (y/n) ")
        burst = 1
        while answer == 'y':
            print("Please click where each sub-burst peak of the next burst is.")
            burst_id = identify_bursts(data, t_samp, favg, peak_bin)
            plt.close('all')

            n_subbursts = len(burst_id.peak_times)
            for sb in range(n_subbursts):
                ind1.append(str(pulse_id) + '-' + str(burst))
                ind2.append('sb' + str(sb + 1))
                DMs.append(dm)

            peak_times.extend(burst_id.peak_times)
            peak_freqs.extend(burst_id.peak_freqs)
            amps.extend(burst_id.peak_amps)

            answer = input("Are there other (separate) bursts in this file? (y/n) ")
            burst += 1

        peak_times = np.array(peak_times) * tavg
        peak_freqs = np.array(peak_freqs)
        amps = np.array(amps) / np.sqrt(tavg)

        peak_freqs = freqs[peak_freqs.astype(np.int)]
        os.chdir('..')
        cols_to_write = pd.MultiIndex.from_tuples([('General','DM'), ('Guesses', 'Amp'),
                                                   ('Guesses', 't_cent'), ('Guesses', 'f_cent')])
        # Note: peak_times is in samples
        for idx1, idx2, dm, amp, time, freq in zip(ind1, ind2, DMs, amps, peak_times, peak_freqs):
            prop_df.loc[(idx1, idx2), cols_to_write] = dm, amp, time, freq

        answer_drops = input("Is the burst affected by dropouts? (y/n) ")
        if answer_drops == 'y':
            prop_df.loc[str(pulse_id), ('General','Dropouts')] = True
        else:
            prop_df.loc[str(pulse_id), ('General','Dropouts')] = False

        # Print from the added pulse to the end
        added_index = (prop_df.index.get_level_values(0)==str(pulse_id)).nonzero()[0][0]
        print(prop_df.iloc[added_index:])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
            prop_df.to_hdf('%s_burst_properties.hdf5' % basename, 'pulses')
