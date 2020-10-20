"""
FOR ARECIBO FRB121102 ANALYSIS

Interactive RFI masker using the filterbank file, and outputs the masked array that can be read into other python scripts. Also does the bandpass correction using import_fil_fits.
Kenzie Nimmo 2020
"""
import sys
#sys.path.insert(1,'~/FRB_filterbank_tools')
import numpy as np
import matplotlib.pyplot as plt
import import_fil_fits
from pylab import *
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import filterbank
import pickle
import os
import re
import optparse
import psrfits
import pandas as pd


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class offpulse(object):
    def __init__(self,filename,gs,dm,AO,in_hdf5_file,pulseindex,tavg):
        self.begin_times=[]
        self.end_times=[]
        self.lines={}

        ax1 = plt.subplot(gs[2]) #dynamic spectrum
        ax2 = plt.subplot(gs[0],sharex=ax1) #profile
        ax3 = plt.subplot(gs[-1],sharey=ax1) #spectrum
        self.ds = ax1
        self.spec = ax3

        self.axes = ax2 # off pulse only necessary for the profile which is in subplot ax2

        self.canvas = ax2.figure.canvas

        if filename.endswith(".fits"):
            fits=psrfits.PsrfitsFile(filename)
            tsamp = fits.specinfo.dt
            arr, startt, peak_bin = import_fil_fits.fits_to_np(filename,dm=dm,maskfile=None,bandpass=False,offpulse=None,AO=True,hdf5=in_hdf5_file,index=pulseindex)

        profile=np.mean(arr,axis=0)
        if tavg>1:
            nsamples = len(profile)
            tavg = float(tavg)
            if (nsamples/tavg)-int(nsamples/tavg)!=0:
                print("The total number of time bins is %s, please choose an tscrunch value that divides the total number of samples."%nsamples)
                sys.exit()
	        else:
                newsamps=nsamples/tavg
                tavg=int(tavg)
                profile=np.mean(profile.reshape(-1, tavg), axis=1)

        self.ax2plot, = ax2.plot(profile, 'k-',alpha=1.0,zorder=1)
        ax2.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')
        ax2.tick_params(axis='x', labelbottom='off', top='off')
        y_range = profile.max() - profile.min()
        ax2.set_ylim(profile.min()-y_range*0.15, profile.max()*1.1)
        ax2.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False, bottom=True, top=True, left=True, right=True)
        ax2.axvline(peak_bin/tavg,color='r')
        fig.add_subplot(ax2)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)
        self.crel = self.canvas.mpl_connect('button_release_event', self.onrel)
        self.keyPress = self.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.keyRelease = self.canvas.mpl_connect('key_release_event', self.onKeyRelease)

        self.data=self.ax2plot.get_data()
        self.profile = self.data[1]
        self.x=False

    def clear_line(self,x):
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
        #if self.ctrlKey == True and self.shiftKey == True:
        if self.x == True:
            tb = get_current_fig_manager().toolbar
            if tb.mode == '':
                x1= event.xdata
                index1=np.int(x1)
                self.begin_times.append(index1)
        if self.x == False:
            return

    def onrel(self, event):
        #if self.ctrlKey == True and self.shiftKey == True:
        if self.x == True:
            tb = get_current_fig_manager().toolbar
            if tb.mode == '':
                x2= event.xdata
                index2=np.int(x2)
                self.end_times.append(index2)
                y_range = self.profile.max() - self.profile.min()
                ymin=self.profile.min()-y_range*0.1
                if self.begin_times[-1] < index2:
                    self.lines['burst'] = self.axes.hlines(y=ymin, xmin=self.begin_times[-1], xmax=index2, lw=10, color='#FF00FF',zorder=0.8)
                else:
                    self.lines['burst'] = self.axes.hlines(y=ymin, xmin=index2, xmax=self.begin_times[-1], lw=10, color='#FF00FF',zorder=0.8)
                plt.draw()
            if self.x == False:
                return



class RFI(object):
    def __init__(self,filename,gs,prof,ds,spec,ithres,ax2,dm,AO,in_hdf5_file,pulseindex,favg,tavg):
        self.begin_chan = []
        self.mask_chan = []
        self.axes = ds # off pulse only necessary for the profile which is in subplot ax2
        self.canvas = ds.figure.canvas
        self.ithres = ithres


        if filename.endswith(".fil"):
            fil=filterbank.filterbank(filename)
            arr = import_fil_fits.filterbank_to_np(filename,dm=dm,maskfile=None,bandpass=False)
            self.total_N=fil.number_of_samples
            self.freqs=fil.frequencies

        if filename.endswith(".fits"):
            fits=psrfits.PsrfitsFile(filename)
            tsamp=fits.specinfo.dt
            arr,startt,peak_bin = import_fil_fits.fits_to_np(filename,dm=dm,maskfile=None,bandpass=False,offpulse=None,AO=True,hdf5=in_hdf5_file,index=pulseindex)
            self.total_N=arr.shape[1]
            self.freqs=fits.frequencies

        if favg>1:
            nchan_tot = arr.shape[0]
            favg=float(favg)
            if (nchan_tot/favg)-int(nchan_tot/favg)!=0:
                print("The total number of channels is %s, please choose an fscrunch value that divides the total number of channels."%nchan_tot)
                sys.exit()
            else:
                newnchan=nchan_tot/favg
                arr=np.array(np.row_stack([np.mean(subint, axis=0) for subint in np.vsplit(arr,newnchan)]))
        if tavg>1:
            nsamples = arr.shape[1]
            tavg = float(tavg)
            if (nsamples/tavg)-int(nsamples/tavg)!=0:
                print("The total number of time bins is %s, please choose an tscrunch value that divides the total number of samples."%nsamples)
                sys.exit()
            else:
                newsamps=nsamples/tavg
                arr=np.array(np.column_stack([np.mean(subint, axis=1) for subint in np.hsplit(arr,newsamps)]))

        spectrum=np.mean(arr,axis=1)
        self.nchans = len(spectrum)
        self.freqbins=np.arange(0,arr.shape[0],1)
        threshold=np.amax(arr)-(np.abs(np.amax(arr)-np.amin(arr))*0.99)

        self.cmap = mpl.cm.binary
        self.ax1 = ds
        self.ax3 = spec
        self.ax2 = ax2
        self.ax2plot = prof
        self.ax1plot = self.ax1.imshow(arr,aspect='auto',vmin=np.amin(arr),vmax=threshold,cmap=self.cmap,origin='lower',interpolation='nearest',picker=True)
        self.cmap.set_over(color='pink')
        self.cmap.set_bad(color='red')
        self.ax1.set_xlim((peak_bin-(0.1/tsamp))/tavg,(peak_bin+(0.1/tsamp))/tavg)

        self.ax3plot, = self.ax3.plot(spectrum, self.freqbins, 'k-',zorder=2)
        self.ax3.tick_params(axis='x', which='both', top='off', bottom='off', labelbottom='off')
        self.ax3.tick_params(axis='y', labelleft='off')
        self.ax3.set_ylim(self.freqbins[0], self.freqbins[-1])
        x_range = spectrum.max() - spectrum.min()
        self.ax3.set_xlim(-x_range/4., x_range*6./5.)

        fig.add_subplot(self.ax1)
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
            arr=self.ax1plot.get_array()
            self.ithres-=0.005
            threshold=np.amax(arr)-(np.abs(np.amax(arr)-np.amin(arr))*self.ithres)
            self.ax1plot.set_clim(vmin=np.amin(arr),vmax=threshold)
            self.cmap.set_over(color='pink')
            plt.draw()


    def onKeyRelease(self, event):
        if event.key == 'x':
            self.x = False
        if event.key == 'r':
            self.r=False


    def onpress(self, event):
        #if self.ctrlKey == True and self.shiftKey == True:
        if self.x == True:
            return
        if self.x == False:
            tb = get_current_fig_manager().toolbar
            if tb.mode == '':
                y1= event.ydata
                arr=self.ax1plot.get_array()
                vmin = np.amin(arr)
                index=find_nearest(self.freqbins,y1)
                self.begin_chan.append(index)

        """
        if self.y == True:
            tb = get_current_fig_manager().toolbar
            if tb.mode == '':
                y_range = self.profile.max() - self.profile.min()
                ymin=self.profile.min()-y_range*0.1
                if begin_times[-1] < end_times[-1]:
                    ax.hlines(y=ymin, xmin=begin_times[-1], xmax=end_times[-1], lw=10, color='#FFFFFF',zorder=1.0)
                else:
                    ax.hlines(y=ymin, xmin=end_times[-1], xmax=begin_times[-1], lw=10, color='#FFFFFF',zorder=1.0)
        """

    def onrel(self, event):
        #if self.ctrlKey == True and self.shiftKey == True:
        if self.x == True:
            return
        if self.x == False:
            tb = get_current_fig_manager().toolbar
            if tb.mode == '':
                y2= event.ydata
                arr=self.ax1plot.get_array()
                vmin = np.amin(arr)
                index2=find_nearest(self.freqbins,y2)
                if self.begin_chan[-1] > index2:
                    arr[index2:self.begin_chan[-1]+1,:]=vmin-100
                else:
                    arr[self.begin_chan[-1]:index2+1,:]=vmin-100
                mask = arr<vmin-50
                arr = np.ma.masked_where(mask==True,arr)
                self.ax1plot.set_data(arr)
                profile = np.mean(arr,axis=0)
                self.ax2plot.set_data(np.arange(0,self.total_N,1),profile)
                self.ax3plot.set_data(np.mean(arr,axis=1),self.freqbins)
                threshold=np.amax(arr)-(np.abs(np.amax(arr)-np.amin(arr))*self.ithres)
                self.ithres-=0.005
                self.ax1plot.set_clim(vmin=np.amin(arr),vmax=threshold)
                spectrum =  np.mean(arr,axis=1)
                self.ax3.set_xlim(np.amin(spectrum),np.amax(spectrum))
                y_range = profile.max() - profile.min()
                self.ax2.set_ylim(profile.min()-y_range*0.1, profile.max()*1.1)

                self.cmap.set_over(color='pink')
                plt.draw()
                if self.begin_chan[-1] > index2:
                    for i in range(len(np.arange(index2,self.begin_chan[-1]+1,1))):
                        self.mask_chan.append(np.arange(index2,self.begin_chan[-1]+1,1)[i])
                else:
                    for i in range(len(np.arange(self.begin_chan[-1],index2+1,1))):
                        self.mask_chan.append(np.arange(self.begin_chan[-1],index2+1,1)[i])

                self.final_spec = np.mean(arr,axis=1)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage='%prog [options] infile_basename', \
                description="Interactive RFI zapper")

    parser.add_option('-f', '--favg', dest='favg', type='int', \
                      help="If -f option is used, frequency averaging is applied using the factor given after -f.", default=1)
    parser.add_option('-t', '--tavg', dest='tavg', type='int', \
                      help="If -t option is used, time averaging is applied using the factor given after -t.", default=1)

    (options, args) = parser.parse_args()
    print(args)
    if len(args)==0:
        parser.print_help()
        sys.exit(1)
    elif len(args)!=1:
        sys.stderr.write("Only one input file must be provided!\n")
    else:
        options.infile = args[-1]


    BASENAME = options.infile
    OUT_DIR = '/data1/nimmo/121102_AO/pipeline_products/%s/pulses/'%BASENAME
    PULSES_TXT = 'pulse_nos.txt'
    in_hdf5_file='../%s.hdf5'%BASENAME
    #find pulses in this dataset
    os.chdir('%s'%OUT_DIR)
    os.system('ls -d */ > %s'%PULSES_TXT)
    pulses=open('%s'%PULSES_TXT)
    pulses_str = []
    pulses_arr = []
    for line in pulses:
        pulses_str.append(line)

    for i in range(len(pulses_str)-1):
        pulses_arr.append(int(pulses_str[i].replace('/\n','')))

    smooth = 10 #smoothing window
    tavg = options.tavg
    favg = options.favg

    for i in range(len(pulses_arr)):
        print("RFI zapping of observation %s, pulse ID %s"%(BASENAME,pulses_arr[i]))

        os.chdir('%s'%pulses_arr[i])
        filename = '%s_%s.fits'%(BASENAME,pulses_arr[i])
        pulses_hdf5 = pd.read_hdf(in_hdf5_file, 'pulses')
        pulses_hdf5=pulses_hdf5.loc[pulses_hdf5['Pulse'] == 0]
        dm=pulses_hdf5.loc[pulses_arr[i],'DM'] #use the DM from detection to de-disperse in this initial stage

        fits=psrfits.PsrfitsFile(filename)
        total_N=fits.specinfo.N
        t_samp=fits.specinfo.dt
        freqs=np.flip(fits.frequencies, 0)
        total_N=int(0.2/t_samp)
        tot_freq = fits.specinfo.num_channels

        rows=2
        cols=2
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, wspace=0., hspace=0., height_ratios=[0.5,]*(rows-1)+[2,], width_ratios=[5,]+[1,]*(cols-1))

        ithres=0.5
        offpulse_prof = offpulse(filename,gs,dm,True,in_hdf5_file,pulses_arr[i],tavg)
        ds = offpulse_prof.ds
        spec = offpulse_prof.spec
        prof = offpulse_prof.ax2plot
        ax2 = offpulse_prof.axes

        #instructions
        print("Click and drag on the dynamic spectrum to identify frequency channels to mask. Hold x and click on the profile and drag to identify where the burst is (no need for complete accuracy as this is so we know the off pulse region). Click y to remove this selection. Click r at any time to refresh the plot and lower the threshold for the pink points (incase there is too much pink points for example).")

        RFImask = RFI(filename,gs,prof,ds,spec,ithres,ax2,dm,True,in_hdf5_file,pulses_arr[i],favg,tavg)
        plt.show()

        begin_times = offpulse_prof.begin_times
        end_times = offpulse_prof.end_times
        if begin_times!=[]:
            if begin_times[-1]<end_times[-1]:
                off_pulse=np.append(np.arange(0,begin_times[-1],1),np.arange(end_times[-1],total_N,1))
            else:
                off_pulse=np.append(np.arange(0,end_times[-1],1),np.arange(begin_times[-1],total_N,1))
        else: print("Warning:  you have not defined the off burst region")

        numchan = np.zeros_like(RFImask.mask_chan)
        numchan+=tot_freq
        mask_chans = np.abs(numchan-np.array(RFImask.mask_chan)-1)

        if begin_times!=[]:
            offpulsefile = '%s_%s_offpulse_time.pkl'%(BASENAME,pulses_arr[i])
            with open(offpulsefile, 'wb') as foff:
                pickle.dump(off_pulse, foff)

        maskfile = '%s_%s_mask.pkl'%(BASENAME,pulses_arr[i])
        with open(maskfile, 'wb') as fmask:
            pickle.dump(mask_chans, fmask)

        #if i == 0:
        #    bursts = {}
        #bursts[str(pulses_arr[i])]={}
        #if smooth == 0:
        #    smooth_val = None
        #else:
        #    smooth_val = smooth
        #if filename.endswith(".fits"):
        #    if options.bandpass ==True:
        #        bursts[str(pulses_arr[i])]['array_corrected'], bursts[str(pulses_arr[i])]['tstart'] =import_fil_fits.fits_to_np(filename,dm=dm, maskfile=maskfile, bandpass=True, offpulse=offpulsefile, smooth_val=smooth_val,AO=True,hdf5=in_hdf5_file,index=pulses_arr[i]) #zapped and bp corrected array
        #    else:
        #        bursts[str(pulses_arr[i])]['array_corrected'], bursts[str(pulses_arr[i])]['tstart']=import_fil_fits.fits_to_np(filename,dm=dm, maskfile=maskfile, bandpass=False, offpulse=offpulsefile, smooth_val=smooth_val,AO=True,hdf5=in_hdf5_file,index=pulses_arr[i]) #zapped array
            #burst['array_uncorrected']=import_fil_fits.fits_to_np(filename,dm=dm,maskfile=None,bandpass=False,offpulse=None,smooth_val=smooth_val,AO=True)
            #burst['undedisp_array']=import_fil_fits.fits_to_np(filename,dm=0, maskfile=maskfile, bandpass=False, offpulse=offpulsefile, smooth_val=smooth_val,AO=False)

        #bursts[str(pulses_arr[i])]['mask']=mask_chans
        #bursts[str(pulses_arr[i])]['t_samp']=t_samp
        #bursts[str(pulses_arr[i])]['tstart']=tstart
        #bursts[str(pulses_arr[i])]['freqs']=freqs
        #bursts[str(pulses_arr[i])]['DM']=dm

        os.chdir('..')

    #df = pd.DataFrame(data=bursts)
    #print(df)
    #df.to_hdf('%s_pulses.hdf5'%BASENAME,'%s'%BASENAME)
