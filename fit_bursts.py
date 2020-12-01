"""
Kenzie Nimmo 2020
"""
import sys
import numpy as np
# sys.path.append('/home/nimmo/AO_burst_properties_pipeline/')
import os
import optparse
import pandas as pd
import import_fil_fits
import burst_2d_Gaussian as fitter

from presto import psrfits

if __name__ == '__main__':
    parser = optparse.OptionParser(usage='%prog [options] infile', description="2D Gaussian fit "
                                   "to FRB data. Input the pickle file output from RFI_masker.py")
    parser.add_option('-t', '--tavg', dest='tavg', type='int', default=1,
                      help="If -t option is used, time averaging is applied using the factor "
                           "given after -t.")
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

    tavg = int(options.tavg)

    # getting the basename of the observation and pulse IDs
    basename = options.infile
    # hdf5 file
    orig_in_hdf5_file = f'../{basename}.hdf5'
    in_hdf5_file = f'{basename}_burst_properties.hdf5'

    out_hdf5_file = in_hdf5_file

    smooth = 7  # smoothing value used for bandpass calibration

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')

    # Get pulses to be processed.
    if options.pulse is not None:
        pulse_ids = [options.pulse]
    else:
        pulse_ids = pulses.index.get_level_values(0).unique()
        if 'Gauss Amp' in pulses.columns:
            not_processed = pulses.loc[(slice(None), ['sb1']), 'Gauss Amp'].isna()
            pulse_ids = pulse_ids[not_processed]

    for pulse_id in pulse_ids:
        print("2D Gaussian fit of observation %s, pulse ID %s" % (basename, pulse_id))
        os.chdir('%s' % pulse_id)
        filename = '%s_%s.fits' % (basename, pulse_id)

        # Get some observation parameters
        fits = psrfits.PsrfitsFile(filename)
        tsamp = fits.specinfo.dt
        freqs = np.flip(fits.frequencies)

        ref_freq = np.max(freqs)

        # Name mask and offpulse file
        maskfile = '%s_%s_mask.pkl' % (basename, pulse_id)
        offpulsefile = '%s_%s_offpulse_time.pkl' % (basename, pulse_id)

        # read in DMs and such
        dm = pulses.loc[(pulse_id, 'sb1'), 'DM']
        amp_guesses = pulses.loc[pulse_id, 'Amp Guess']
        time_guesses = pulses.loc[pulse_id, 'Peak Time Guess']

        begin_time = int(time_guesses.min() - int(15e-3 / (tavg * tsamp)))
        end_time = int(time_guesses.max() + int(15e-3 / (tavg * tsamp)))

        waterfall, t, peak_bin = import_fil_fits.fits_to_np(
            filename, dm=dm, maskfile=maskfile, bandpass=True, offpulse=offpulsefile, AO=True,
            smooth_val=smooth, hdf5=orig_in_hdf5_file, index=pulse_id, tavg=tavg)
        waterfall = waterfall[:, begin_time:end_time]

        maskbool = waterfall.mask
        maskbool = maskbool.astype('uint8')
        maskbool -= 1
        maskbool = np.abs(maskbool)

        waterfall = waterfall.data * maskbool

        time_guesses -= begin_time

        timebins = np.arange(waterfall.shape[1])
        freqbins = np.arange(waterfall.shape[0])

        print(time_guesses)
        print(amp_guesses)
        print(pulses)

        n_sbs = pulses.loc[pulse_id].shape[0]
        freq_peak_guess = n_sbs * [int(512 / 2.)]
        freq_std_guess = n_sbs * [int(512 / 4.)]
        t_std_guess = n_sbs * [2e-3 / (tsamp * tavg)]
        while True:
            model = fitter.gen_Gauss2D_model(time_guesses, amp_guesses, f0=freq_peak_guess,
                                             bw=freq_std_guess, dt=t_std_guess)
            bestfit, fitLM = fitter.fit_Gauss2D_model(waterfall, timebins, freqbins, model)
            bestfit_params, bestfit_errors = fitter.report_Gauss_parameters(bestfit, fitLM,
                                                                            verbose=True)
            fitter.plot_burst_windows(timebins, freqbins, waterfall, bestfit, ncontour=8,
                                      res_plot=True)  # diagnostic plots

            answer = input("Are you happy with the fit y/n?")
            if answer == 'y':
                break
            if answer == 'n':
                guessvals = []
                while len(guessvals) != n_sbs*3:
                    secondanswer = input("Give the intial guesses in samples and channels in the"
                                         "form t_std_sb1,t_std_sb2,...,f_peak_sb1,...,f_std_sb1,"
                                         f"... with {n_sbs*3} guesses in total.")
                    guessvals = [int(x.strip()) for x in secondanswer.split(',')]
                t_std_guess = guessvals[0 : n_sbs]
                freq_peak_guess = guessvals[n_sbs : 2*n_sbs]
                freq_std_guess = guessvals[2*n_sbs:]
            else:
                print("Please provide an answer y or n.")

        os.chdir('..')

        pulses.loc[pulse_id, 'Gauss Amp'] = bestfit_params[:, 0]
        pulses.loc[pulse_id, 'Gauss Amp e'] = bestfit_errors[:, 0]
        pulses.loc[pulse_id, 't_cent / s'] = bestfit_params[:, 1] * tsamp * tavg
        pulses.loc[pulse_id, 't_cent_e / s'] = bestfit_errors[:, 1] * tsamp * tavg
        pulses.loc[pulse_id, 't_std / s'] = bestfit_params[:, 3] * tsamp * tavg
        pulses.loc[pulse_id, 't_std_e / s'] = bestfit_errors[:, 3] * tsamp * tavg
        pulses.loc[pulse_id, 'f_cent / MHz'] = bestfit_params[:, 2] * 1.5625
        pulses.loc[pulse_id, 'f_cent_e / MHz'] = bestfit_errors[:, 2] * 1.5625
        pulses.loc[pulse_id, 'f_std / MHz'] = bestfit_params[:, 4] * 1.5625
        pulses.loc[pulse_id, 'f_std_e / MHz'] = bestfit_errors[:, 4] * 1.5625
        pulses.loc[pulse_id, 'f_ref / MHz'] = ref_freq
        pulses.loc[pulse_id, 'Gauss Angle'] = bestfit_params[:, 5]
        pulses.loc[pulse_id, 'Gauss Angle e'] = bestfit_errors[:, 5]

        pulses.to_hdf(out_hdf5_file, 'pulses')
        print(pulses)
