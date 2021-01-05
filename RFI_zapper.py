"""
FOR ARECIBO FRB121102 ANALYSIS

Interactive RFI masker using the filterbank file, and outputs the masked array that can be read
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
from presto import filterbank
import pickle
import os
#import re
import optparse
#import psrfits_pipe as psrfits
from presto import psrfits
import pandas as pd
import warnings

from scipy.stats import normaltest


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class identify_bursts(object):
    def __init__(self, filename, gs, tavg, dm, in_hdf5_file, pulseindex, maskfile, offpulsefile):
        self.peak_times = []
        self.peak_amps = []

        ax1 = plt.subplot(gs[0])  # profile
        self.axes = ax1
        self.canvas = ax1.figure.canvas

        if filename.endswith(".fits"):
            fits = psrfits.PsrfitsFile(filename)
            tsamp = fits.specinfo.dt
            self.arr, startt, peak_bin = import_fil_fits.fits_to_np(
                filename, dm=dm, maskfile=maskfile, AO=True,
                hdf5=in_hdf5_file, index=pulseindex, plot=False, tavg=tavg
                )
            self.peak_bin = peak_bin
            self.tsamp = tsamp

        arr = self.arr
        profile = np.sum(arr, axis=0)

        # offpulse times
        with open(offpulsefile, 'rb') as f:
            offtimes = pickle.load(f)
        if tavg != 1:
            # Transform the offtimes into the downsampled shape.
            offlen = offtimes.shape[0]
            offtimes = offtimes.reshape(offlen//tavg, tavg).all(axis=1)

        offprof = profile[offtimes]
        # convert to S/N
        profile -= np.mean(offprof)
        offprof -= np.mean(offprof)
        profile /= np.std(offprof)

        # convert arr to S/N
        offarr = arr[:, offtimes]
        arr -= np.mean(offarr)
        offarr -= np.mean(offarr)
        arr /= np.std(offarr)

        self.arr = arr

        self.ax1plot, = ax1.plot(profile, 'k-', alpha=1.0, zorder=1)
        y_range = profile.max() - profile.min()
        ax1.set_ylim(profile.min() - y_range * 0.15, profile.max() + y_range * 0.1)
        ax1.axvline((peak_bin) - (5e-3 / (tavg * tsamp)), color='r', linestyle='--')
        ax1.axvline((peak_bin) + (5e-3 / (tavg * tsamp)), color='r', linestyle='--')
        fig.add_subplot(ax1)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        indices = np.array(self.peak_times).argsort()
        self.peak_times = list(np.array(self.peak_times)[indices])
        self.peak_amps = list(np.array(self.peak_amps)[indices])

    def onpress(self, event):
        tb = plt.get_current_fig_manager().toolbar
        if tb.mode == '':
            y1 = event.ydata
            x1 = event.xdata
            self.peak_times.append(x1)
            maxval = np.max(self.arr[:, int(x1)])
            meanval = np.mean(self.arr[:, int(x1)])
            self.peak_amps.append(meanval)  #maxval
            self.axes.scatter(x1, y1, lw=3, color='r', marker='x', s=100, zorder=10)
            plt.draw()


class offpulse(object):
    def __init__(self, filename, gs, dm, AO, in_hdf5_file, pulseindex, tavg, initial_mask=None):
        self.begin_times = []
        self.end_times = []
        self.lines = {}

        ax1 = plt.subplot(gs[3])  # dynamic spectrum
        ax2 = plt.subplot(gs[0], sharex=ax1)  # profile
        ax3 = plt.subplot(gs[-1], sharey=ax1)  # spectrum
        ax4 = plt.subplot(gs[-2], sharey=ax1)
        self.ds = ax1
        self.spec = ax3
        self.stat = ax4  # To plot channel statistics

        self.axes = ax2  # off pulse only necessary for the profile which is in subplot ax2

        self.canvas = ax2.figure.canvas

        if filename.endswith(".fits"):
            fits = psrfits.PsrfitsFile(filename)
            tsamp = fits.specinfo.dt
            #if initial_mask is None:
            #    arr, startt, peak_bin = import_fil_fits.fits_to_np(filename, dm=dm, maskfile=None,
            #                                                       bandpass=False, offpulse=None,
            #                                                       AO=True, hdf5=in_hdf5_file,
            #                                                       index=pulseindex, tavg=tavg)
            #else:
            arr, startt, peak_bin = import_fil_fits.fits_to_np(
                filename, dm=dm, maskfile=initial_mask,
                AO=True, hdf5=in_hdf5_file, index=pulseindex, tavg=tavg
                )
            self.peak_bin = peak_bin
            self.tsamp = tsamp

        profile = np.mean(arr, axis=0)

        self.ax2plot, = ax2.plot(profile, 'k-', alpha=1.0, zorder=1)
        ax2.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        ax2.tick_params(axis='x', labelbottom='off', top='off')
        y_range = profile.max() - profile.min()
        ax2.set_ylim(profile.min() - y_range * 0.15, profile.max() + y_range * 0.1)
        ax2.tick_params(labelbottom=False, labeltop=False, labelleft=False,
                        labelright=False, bottom=True, top=True, left=True, right=True)
        ax2.axvline((peak_bin) - (5e-3 / (tavg * tsamp)), color='r', linestyle='--')
        ax2.axvline((peak_bin) + (5e-3 / (tavg * tsamp)), color='r', linestyle='--')
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
        if event.key == 'x':
            self.x = True
        if event.key == 'y':
            if self.lines['burst']:
                self.clear_line('burst')
                plt.draw()

    def onKeyRelease(self, event):
        if event.key == 'x':
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
                y_range = self.profile.max() - self.profile.min()
                ymin = self.profile.min()
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
    def __init__(self, filename, gs, prof, ds, spec, stat, ithres, ax2, dm, AO,
                 in_hdf5_file, pulseindex, favg, tavg, initial_mask=None):
        self.begin_chan = []
        self.mask_chan = []
        if initial_mask is not None:
            initial = pickle.load(open(initial_mask, 'rb'))
            self.mask_chan = [int(x) for x in initial]
        self.axes = ds  # off pulse only necessary for the profile which is in subplot ax2
        self.canvas = ds.figure.canvas
        self.ithres = ithres

        if filename.endswith(".fil"):
            fil = filterbank.filterbank(filename)
            arr = import_fil_fits.filterbank_to_np(filename, dm=dm, maskfile=None)
            self.total_N = fil.number_of_samples
            self.freqs = fil.frequencies

        if filename.endswith(".fits"):
            fits = psrfits.PsrfitsFile(filename)
            tsamp = fits.specinfo.dt
            #if initial_mask is None:
            #    arr, startt, peak_bin = import_fil_fits.fits_to_np(
            #        filename, dm=dm, maskfile=None, bandpass=False, offpulse=None, AO=True,
            #        hdf5=in_hdf5_file, index=pulseindex, tavg=tavg
            #        )
            #else:
            arr, startt, peak_bin = import_fil_fits.fits_to_np(
                filename, dm=dm, maskfile=initial_mask,
                AO=True, hdf5=in_hdf5_file, index=pulseindex, tavg=tavg
                )
            self.total_N = arr.shape[1]
            self.freqs = fits.frequencies
            self.nchan = len(self.freqs)

        if favg > 1:
            nchan_tot = arr.shape[0]
            favg = float(favg)
            if (nchan_tot / favg) - int(nchan_tot / favg) != 0:
                print(f"The total number of channels is {nchan_tot}, please choose an fscrunch "
                      + "value that divides the total number of channels."
                      )
                sys.exit()
            else:
                newnchan = nchan_tot / favg
                arr = np.array(np.row_stack([np.mean(subint, axis=0)
                                             for subint in np.vsplit(arr, newnchan)]))

        spectrum = np.mean(arr, axis=1)
        self.nchans = len(spectrum)
        self.freqbins = np.arange(0, arr.shape[0], 1)
        threshold = np.amax(arr) - (np.abs(np.amax(arr) - np.amin(arr)) * 0.99)

        self.cmap = mpl.cm.binary
        self.ax1 = ds
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
                                       interpolation='nearest',
                                       picker=True)
        self.cmap.set_over(color='pink')
        self.cmap.set_bad(color='red')
        #self.ax1.set_xlim((peak_bin - (50e-3 / (tsamp * tavg))),
        #                  (peak_bin + (50e-3 / (tsamp * tavg))))

        self.ax3plot, = self.ax3.plot(spectrum, self.freqbins, 'k-', zorder=2)
        self.ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
        self.ax3.tick_params(axis='y', labelleft='off')
        self.ax3.set_ylim(self.freqbins[-1], self.freqbins[0])
        x_range = spectrum.max() - spectrum.min()
        self.ax3.set_xlim(-x_range / 4., x_range * 6. / 5.)

        # Plot channel statistics
        normal_deviation = normaltest(arr, axis=1)[0]
        self.ax4plot, = self.ax4.plot(normal_deviation, self.freqbins, 'k-', zorder=2)
        self.ax4.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
        self.ax4.tick_params(axis='y', labelleft='off')
        self.ax4.set_ylim(self.freqbins[-1], self.freqbins[0])
        self.ax4.set_xlim(-x_range / 4., x_range * 6. / 5.)

        fig.add_subplot(self.ax1)
        fig.add_subplot(self.ax4)
        fig.add_subplot(self.ax3)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.crel = self.canvas.mpl_connect('button_release_event', self.onrel)
        self.keyPress = self.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.keyRelease = self.canvas.mpl_connect('key_release_event', self.onKeyRelease)
        self.x = False
        self.r = False

    def onKeyPress(self, event):
        if event.key == 'x':
            self.x = True
        if event.key == 'r':
            self.r = True
            arr = self.ax1plot.get_array()
            self.ithres -= 0.005
            threshold = np.amax(arr) - (np.abs(np.amax(arr) - np.amin(arr)) * self.ithres)
            self.ax1plot.set_clim(vmin=np.amin(arr), vmax=threshold)
            self.cmap.set_over(color='pink')
            plt.draw()

    def onKeyRelease(self, event):
        if event.key == 'x':
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
                arr = self.ax1plot.get_array()
                vmin = np.amin(arr)
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
                arr = self.ax1plot.get_array()
                vmin = np.amin(arr)
                index2 = find_nearest(self.freqbins, y2)
                if self.begin_chan[-1] > index2:
                    arr[index2:self.begin_chan[-1] + 1, :] = vmin - 100
                else:
                    arr[self.begin_chan[-1]:index2 + 1, :] = vmin - 100
                mask = arr < vmin - 50
                arr = np.ma.masked_where(mask, arr)
                self.ax1plot.set_data(arr)
                profile = np.mean(arr, axis=0)
                self.ax2plot.set_data(np.arange(0, self.total_N, 1), profile)
                spectrum = np.mean(arr, axis=1)
                self.ax3plot.set_data(spectrum, self.freqbins)
                normal_deviation = normaltest(arr, axis=1)[0]
                normal_deviation = np.ma.masked_where(mask[:, 0], normal_deviation)
                self.ax4plot.set_data(normal_deviation, self.freqbins)

                threshold = np.amax(arr) - (np.abs(np.amax(arr) - np.amin(arr)) * self.ithres)
                self.ithres -= 0.005
                self.ax1plot.set_clim(vmin=np.amin(arr), vmax=threshold)

                self.ax3.set_xlim(np.amin(spectrum), np.amax(spectrum))
                self.ax4.set_xlim(np.amin(normal_deviation), np.amax(normal_deviation))
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
                      help="If -f option is used, frequency averaging is applied using the "
                           "factor given after -f.")
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

    if options.mask is not None:
        initial_mask = options.mask
    else:
        initial_mask = None

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

# =============================================================================
#
#     pulses_str = []
#     pulses_arr = []
#     for line in pulses:
#         pulses_str.append(line)
#     print(pulses_str)
#
#     for i in range(len(pulses_str) - 1):
#         pulses_arr.append(int(pulses_str[i].replace('/\n', '')))
# =============================================================================

    smooth = None  # smoothing window
    tavg = options.tavg
    favg = options.favg

    prop_file = f'{basename}_burst_properties.hdf5'
    if os.path.isfile(prop_file):
        prop_df = pd.read_hdf(prop_file, 'pulses')
    else:
        index = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['pulse_id', 'subburst'])
        cols = pd.MultiIndex.from_tuples([('General','DM'),
                                          ('Guesses', 'Amp'),
                                          ('Guesses', 't_cent')])
        prop_df = pd.DataFrame(index=index, columns=cols, dtype=np.float)

    print(prop_df)
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

        #total_N=fits.specinfo.N / tavg
        t_samp = fits.specinfo.dt
        freqs = fits.frequencies[::-1]
        tot_freq = fits.specinfo.num_channels
        #total_N = int(100e-3 / (t_samp * tavg))

        rows = 2
        cols = 3
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(rows, cols, wspace=0., hspace=0.,
                               height_ratios=[0.5,]*(rows-1) + [2,],
                               width_ratios=[5,] + [1,]*(cols-1))

        ithres = 0.5
        offpulse_prof = offpulse(filename, gs, dm, True, pulses_hdf5, pulse_id, tavg,
                                 initial_mask=initial_mask)
        ds = offpulse_prof.ds
        spec = offpulse_prof.spec
        prof = offpulse_prof.ax2plot
        ax2 = offpulse_prof.axes
        stat = offpulse_prof.stat

        # instructions
        print("Click and drag on the dynamic spectrum to identify frequency channels to mask.\n"
              "Hold x and click on the profile and drag to identify where the burst is (no "
              "need for complete accuracy as this is so we know the off pulse region).\n"
              "Click y to remove this selection.\n"
              "Click r at any time to refresh the plot and lower the threshold for the pink "
              "points (incase there is too much pink points for example).")

        RFImask = RFI(filename, gs, prof, ds, spec, stat, ithres, ax2, dm, True, in_hdf5_file,
                      pulse_id, favg, tavg, initial_mask=initial_mask)
        plt.show()

        profile = RFImask.final_prof
        peak_bin = offpulse_prof.peak_bin
        tsamp = offpulse_prof.tsamp

        begin_times = offpulse_prof.begin_times
        end_times = offpulse_prof.end_times
        total_N = len(profile)
        # Save the offpulse as if it was not downsampled
        if tavg != 1:
            total_N *= tavg
            begin_times *= tavg
            end_times *= tavg

        off_pulse = np.ones(total_N, dtype=np.bool)
        if begin_times == []:
            raise ValueError("Warning: you have not defined the off burst region")
        if len(begin_times) != len(end_times):
            raise ValueError("The number of begin and end times doesn't match, make sure to hold "
                             "x")
        else:
            for begin_time, end_time in zip(begin_times, end_times):
                if begin_time < end_time:
                    #off_pulse = np.append(np.arange(0, begin_times[-1], 1),
                    #                      np.arange(end_times[-1], total_N - 1, 1))
                    off_pulse[begin_time:end_time] = False
                else:
                    off_pulse[end_time:begin_time:-1] = False
                    #off_pulse = np.append(np.arange(0, end_times[-1], 1),
                    #                      np.arange(begin_times[-1], total_N - 1, 1))


        #numchan = np.zeros_like(RFImask.mask_chan)
        #numchan+=tot_freq
        mask_chans = np.abs(np.array(RFImask.mask_chan))

        if begin_times != []:
            offpulsefile = '%s_%s_offpulse_time.pkl' % (basename, pulse_id)
            with open(offpulsefile, 'wb') as foff:
                pickle.dump(off_pulse, foff)

        maskfile = '%s_%s_mask.pkl' % (basename, pulse_id)
        with open(maskfile, 'wb') as fmask:
            pickle.dump(mask_chans, fmask)

        # Are there sub-bursts?
        answer = input("Does the burst have multiple components? (y/n) ")
        if answer == 'y':
            answer_sub = int(input("How many? (integer) "))
        if answer == 'n':
            answer_sub = 1

        DMs = answer_sub*[dm]
        for j in range(answer_sub):
            ind1.append(str(pulse_id))
            ind2.append('sb' + str(j + 1))

        # Are there other bursts in this file?
        answer = input("Are there other (separate) bursts in this file? (y/n) ")
        if answer == 'y':
            answer_burst = int(input("How many? (integer) "))
            answer_burst_sub = input("Does this (these) burst(s) have multiple components? (y/n) ")
            if answer_burst_sub == 'y':
                sub_components_other_bursts = input(
                    "Give the number of components per burst (in order of arrival time -- "
                    "excluding main burst), separated by commas e.g. 3,2,1,3 ")
                n_subbs = [int(x.strip()) for x in sub_components_other_bursts.split(',')]

            if answer_burst_sub == 'n':
                n_subbs = answer_burst*[1]

        if answer == 'n':
            answer_burst = 0

        for bu in range(answer_burst):
            #os.system('mkdir ../%s' % str(pulse_id) + '-' + str(bu + 1))
            #os.system('cp ./%s ../%s/%s_%s.fits' % (
            #    filename, str(pulse_id) + '-' + str(bu + 1), basename,
            #    str(pulse_id) + '-' + str(bu + 1)))
            #os.system('cp ./%s_%s_mask.pkl ../%s/%s_%s_mask.pkl' % (
            #    basename, pulse_id, str(pulse_id) + '-' + str(bu + 1), basename,
            #    str(pulse_id) + '-' + str(bu + 1)))
            for sb in range(n_subbs[bu]):
                ind1.append(str(pulse_id) + '-' + str(bu + 1))
                ind2.append('sb' + str(sb + 1))
                DMs.append(dm)

        rows = 1
        cols = 1
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(rows, cols)

        print("Please click where each sub-burst peak of the main burst is.")
        burst_id = identify_bursts(filename, gs, tavg, dm, in_hdf5_file,
                                   pulse_id, maskfile, offpulsefile)
        plt.show()

        main_burst_expected = answer_sub

        while len(burst_id.peak_times) != main_burst_expected:
            fig = plt.figure(figsize=(10, 10))
            gs = gridspec.GridSpec(rows, cols)
            print("The number of selections did not match the total number of components. "
                  "Please try again.")
            burst_id = identify_bursts(filename, gs, tavg, dm,
                                       in_hdf5_file, pulse_id, maskfile, offpulsefile)
            plt.show()

        peak_times = np.array(burst_id.peak_times) * tavg
        amps = burst_id.peak_amps

        if answer == 'y':
            for other_bursts in range(answer_burst):
                fig = plt.figure(figsize=(10, 10))
                gs = gridspec.GridSpec(rows, cols)
                print("Please click where each sub-burst peak of the other bursts are (ordered "
                      "same as before).")
                burst_id = identify_bursts(filename, gs, tavg, dm,
                                           in_hdf5_file, pulse_id, maskfile, offpulsefile)
                plt.show()
                other_burst_expected = n_subbs[other_bursts]

                while len(burst_id.peak_times) != other_burst_expected:
                    fig = plt.figure(figsize=(10, 10))
                    gs = gridspec.GridSpec(rows, cols)
                    print("The number of selections did not match the total number of "
                          "components. Please try again.")
                    burst_id = identify_bursts(filename, gs, tavg, dm, in_hdf5_file,
                                               pulse_id, maskfile, offpulsefile)
                    plt.show()

                # remove the additional burst from the offpulsetimes
                begin_eb = np.min(burst_id.peak_times) - (10e-3 / (tsamp * tavg))
                end_eb = np.max(burst_id.peak_times) + (10e-3 / (tsamp * tavg))

                off_pulse = np.array(off_pulse)
                off_pulse = off_pulse[(off_pulse < begin_eb) | (off_pulse > end_eb)]

                peak_times = np.append(peak_times, np.array(burst_id.peak_times) * tavg)
                amps.extend(burst_id.peak_amps)

            offpulsefile = '%s_%s_offpulse_time.pkl' % (basename, pulse_id)
            with open(offpulsefile, 'wb') as foff:
                pickle.dump(off_pulse, foff)

            #os.system('cp ./%s_%s_offpulse_time.pkl ../%s/%s_%s_offpulse_time.pkl' %
            #          (basename, pulse_id, str(pulse_id) + '-' + str(other_bursts + 1), basename,
            #           str(pulse_id) + '-' + str(other_bursts + 1)))

        os.chdir('..')
        cols_to_write = pd.MultiIndex.from_tuples([('General','DM'), ('Guesses', 'Amp'),
                                                   ('Guesses', 't_cent')])
        # Note: peak_times is in samples
        for idx1, idx2, dm, amp, time in zip(ind1, ind2, DMs, amps, peak_times):
            prop_df.loc[(idx1, idx2), cols_to_write] = dm, amp, time

        print(prop_df)
        prop_df.to_hdf('%s_burst_properties.hdf5' % basename, 'pulses')
