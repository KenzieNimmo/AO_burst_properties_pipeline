"""
Kenzie Nimmo 2020
"""
import sys
import numpy as np
import import_fil_fits
sys.path.append('/home/nimmo/AO_burst_properties_pipeline/')
from fit_repeater_bursts import fit_my_smudge
import filterbank
import os
import pickle
import matplotlib.pyplot as plt
import re
import optparse
import pandas as pd
import psrfits

if __name__ == '__main__':
    parser = optparse.OptionParser(usage='%prog [options] infile', \
                description="2D Gaussian fit to FRB data. Input the pickle file output from RFI_masker.py")
    #parser.add_option('-g', '--guess', dest='guess', type='string', \
                      #help="Guess for gaussian fit. time_peak:time_width:freq_peak:freq_width.", default=None)
    #parser.add_option('-u', '--uncorr', dest='uncorr', action="store_true", \
                      #help="If -u option is used, use the uncorrected array (otherwise use the masked+bandpass corrected array).", default=False)
    #parser.add_option('-d', '--dm', dest='dm', type='float',help="Dispersion measure.", default=None)
    parser.add_option('-f', '--FRBname', dest='FRBname', type='string',help="FRB R name.", default=None)
    parser.add_option('-t', '--telescopename', dest='telescopename', type='string',help="Telescope used (Eff, CHIME, DSS43).", default=None)
    (options, args) = parser.parse_args()

    if len(args)==0:
        parser.print_help()
        sys.exit(1)
    elif len(args)!=1:
        sys.stderr.write("Only one input file must be provided!\n")
    else:
        options.infile = args[-1]

    #getting the basename of the observation and pulse IDs
    BASENAME = options.infile
    PULSES_TXT = 'pulse_nos.txt'
    #hdf5 file
    orig_in_hdf5_file='../%s.hdf5'%BASENAME
    in_hdf5_file='%s_burst_properties.hdf5'%BASENAME
    out_hdf5_file=in_hdf5_file

    smooth=7 #smoothing value used for bandpass calibration

    pulses=open('%s'%PULSES_TXT)
    pulses_str = []
    pulses_arr = []
    for line in pulses:
        pulses_str.append(line)

    for i in range(len(pulses_str)-1):
        pulses_arr.append(int(pulses_str[i].replace('/\n','')))

    
    t_cent = []
    t_cent_e = []
    t_width = []
    t_width_e = []
    f_cent = []
    f_cent_e = []
    f_width = []
    f_width_e = []
    f_ref = []

    for i in range(len(pulses_arr)):
        print("2D Gaussian fit of observation %s, pulse ID %s"%(BASENAME,pulses_arr[i]))
        os.chdir('%s'%pulses_arr[i])
        filename = '%s_%s.fits'%(BASENAME,pulses_arr[i])
        fits=psrfits.PsrfitsFile(filename)
        tsamp=fits.specinfo.dt
        freqs=np.flip(fits.frequencies)

        ref_freq = np.max(freqs)
        f_ref.append(ref_freq)

        #read in mask file
        maskfile = '%s_%s_mask.pkl'%(BASENAME,pulses_arr[i])
        #read in offpulse file
        offpulsefile = '%s_%s_offpulse_time.pkl'%(BASENAME,pulses_arr[i])

        #read in DMs
        pulses = pd.read_hdf('../'+in_hdf5_file, 'pulses')
        dm=pulses.loc[pulses_arr[i],'DM']

        waterfall,t,peak_bin=import_fil_fits.fits_to_np(filename,dm=dm,maskfile=maskfile,bandpass=True,offpulse=offpulsefile,AO=True,smooth_val=smooth,hdf5=orig_in_hdf5_file,index=pulses_arr[i])


        prof = np.mean(waterfall,axis=0)
        spec = np.mean(waterfall,axis=1)

        stimes = np.linspace(0,waterfall.shape[1],waterfall.shape[1])
        stimes*=tsamp
        #guess=[dmax, xmax, ymax, xwid, ywid, 0]
        guess = []
        for g in range(10):
            times,fit, bin_center = fit_my_smudge(waterfall, stimes, freqs, guess=guess, doplot=True, basename=BASENAME)
            answer=raw_input("Are you happy with the fit y/n?")
            if answer == 'y':
                break
            if answer == 'n':
                secondanswer=raw_input("Give the intial guesses in the form xmax,ymax,xwid,ywid")
                guessvals = [int(x.strip()) for x in secondanswer.split(',')]
                guess = [50, guessvals[0], guessvals[1], guessvals[2], guessvals[3], 0]
            else: print("Please provide an answer y or n.")

        os.chdir('..')

        t_cent.append(fit[1])
        t_cent_e.append(fit[2])
        t_width.append(fit[5])
        t_width_e.append(fit[6])
        f_cent.append(fit[3])
        f_cent_e.append(fit[4])
        f_width.append(fit[7])
        f_width_e.append(fit[8])

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')
    fit_numbers = {'t_cent [s]':t_cent,'t_cent_e [s]':t_cent_e,'t_width [s]':t_width,'t_width_e [s]':t_width_e,'f_cent [MHz]':f_cent,'f_cent_e [MHz]':f_cent_e,'f_width [MHz]':f_width,'f_width_e [MHz]':f_width_e,'f_ref [MHz]':f_ref}
    df = pd.DataFrame(fit_numbers,index=pulses_arr)
    pulses=pulses.join(df)
    pulses.to_hdf(out_hdf5_file,'pulses')
