"""

AO ANALYSIS FRB121102

Python tool to feed chopped fits file into DM_phase to determine the
structure-optimising dispersion measure.

From the README: it is possible to run the function get_DM on a 2D numpy array
representing the pulse waterfall.

Kenzie Nimmo 2020
"""
import DM_phase
import optparse
import import_fil_fits
import numpy as np
import psrfits
import sys
import os
import pandas as pd


if __name__ == '__main__':
    parser = optparse.OptionParser(usage='%prog [options] infile_basename', \
                description="DM_phase")

    parser.add_option('-d', '--DM_s', dest='DM_s', type='float', \
                      help="-d/--DM_s to define the start DM for the search. Default 560-10. ", default=550.)
    parser.add_option('-e', '--DM_e', dest='DM_e', type='float', \
                      help="-e/--DM_e to define the end DM for the search. Default 560+10. ", default=570.)
    parser.add_option('-s', '--DM_step', dest='DM_step', type='float', \
                      help="-s/--DM_step to define the steps in DM for the search. Default 0.1 ", default=0.1)
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
    OUT_DIR = '/data1/nimmo/121102_AO/pipeline_products/%s/pulses/'%BASENAME
    PULSES_TXT = 'pulse_nos.txt'
    #hdf5 file
    in_hdf5_file='../%s.hdf5'%BASENAME
    out_hdf5_file='%s_burst_properties.hdf5'%BASENAME

    os.chdir('%s'%OUT_DIR)
    pulses=open('%s'%PULSES_TXT)
    pulses_str = []
    pulses_arr = []
    for line in pulses:
        pulses_str.append(line)

    for i in range(len(pulses_str)-1):
        pulses_arr.append(int(pulses_str[i].replace('/\n','')))

    #hardcoded for testing
    pulses_arr=[8898,9362]

    DMperpulse=np.zeros_like(pulses_arr)
    DMsigperpulse=np.zeros_like(pulses_arr)
    tstarts = np.zeros_like(pulses_arr)
    for i in range(len(pulses_arr)):
        print("DM_phase of observation %s, pulse ID %s"%(BASENAME,pulses_arr[i]))
        os.chdir('%s'%pulses_arr[i])
        filename = '%s_%s.fits'%(BASENAME,pulses_arr[i])

        #read in mask file
        maskfile = '%s_%s_mask.pkl'%(BASENAME,pulses_arr[i])

        waterfall,t,p=import_fil_fits.fits_to_np(filename,dm=0,maskfile=maskfile,bandpass=False,offpulse=None,AO=True,hdf5=in_hdf5_file,index=pulses_arr[i])
        DM_list = np.arange(options.DM_s,options.DM_e,options.DM_step)
        fits=psrfits.PsrfitsFile(filename)
        t_res=fits.specinfo.dt
        f_channels = np.flip(fits.frequencies)

        DM, DM_std = DM_phase.get_DM(waterfall,DM_list,t_res,f_channels,ref_freq="top", \
            manual_cutoff=False, manual_bandwidth=False, diagnostic_plots=True, \
            fname="test", no_plots=False)

        DMperpulse[i]=DM
        DMsigperpulse[i]=DM_std
        tstarts[i]=t

        os.chdir('..')
        print("DM of %s pulse %s is %s +- %s"%(BASENAME,pulses_arr[i],DM,DM_std))

    #write out the pandas dataframe containing the DMs.
    DMs={'idx':pulses_arr,'DM':DMperpulse,'DMstd':DMsigperpulse,'tstart of fits [MJD]':tstarts}
    df = pd.DataFrame(DMs,index=pulses_arr)
    df.to_hdf(out_hdf5_file,'pulses')
