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
    pulses_txt = 'pulse_nos.txt'
    # hdf5 file
    orig_in_hdf5_file = f'../{basename}.hdf5'
    in_hdf5_file = f'{basename}_burst_properties.hdf5'

    out_hdf5_file = in_hdf5_file

    smooth = 7  # smoothing value used for bandpass calibration

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')
    indices = pulses.index.values.tolist()
    bursts, sbs = list(zip(*indices))
    bursts = np.array(bursts)
    bursts_unique, number_of_sbs = np.unique(bursts, return_counts=True)

    pulses_arr = [str(x) for x in bursts_unique]

    t_cent = []
    t_cent_e = []
    t_width = []
    t_width_e = []
    f_cent = []
    f_cent_e = []
    f_width = []
    f_width_e = []
    f_ref = []
    gaus_angle = []
    amplitudes = []

    for i in range(len(pulses_arr)):
        # pulses_arr[i]=str(pulses_arr[i])
        print("2D Gaussian fit of observation %s, pulse ID %s" % (basename, pulses_arr[i]))
        os.chdir('%s' % pulses_arr[i])
        filename = '%s_%s.fits' % (basename, pulses_arr[i])
        fits = psrfits.PsrfitsFile(filename)
        tsamp = fits.specinfo.dt
        freqs = np.flip(fits.frequencies)

        ref_freq = np.max(freqs)
        f_ref.append(ref_freq)

        # read in mask file
        maskfile = '%s_%s_mask.pkl' % (basename, pulses_arr[i])
        # read in offpulse file
        offpulsefile = '%s_%s_offpulse_time.pkl' % (basename, pulses_arr[i])

        # read in DMs
        current_burst_id = np.where(bursts == pulses_arr[i])[0]
        amp_guesses = []
        time_guesses = []
        for j in range(len(current_burst_id)):
            amp_guesses = np.append(
                amp_guesses, pulses.loc[indices[current_burst_id[j]], 'Amp Guess'])
            time_guesses = np.append(
                time_guesses, pulses.loc[indices[current_burst_id[j]], 'Peak Time Guess'])

        amp_guesses = list(amp_guesses)
        time_guesses = list(time_guesses)
        time_guesses = np.array(time_guesses)

        begin_time = int(time_guesses[0] - int(15e-3 / (tavg * tsamp)))
        end_time = int(time_guesses[-1] + int(15e-3 / (tavg * tsamp)))

        dm = pulses.loc[(pulses_arr[i], 'sb1'), 'DM']

        waterfall, t, peak_bin = import_fil_fits.fits_to_np(
            filename, dm=dm, maskfile=maskfile, bandpass=True, offpulse=offpulsefile, AO=True,
            smooth_val=smooth, hdf5=orig_in_hdf5_file, index=pulses_arr[i], tavg=tavg)
        waterfall = waterfall[:, begin_time:end_time]

        maskbool = waterfall.mask
        maskbool = maskbool.astype('uint8')
        maskbool -= 1
        maskbool = np.abs(maskbool)

        waterfall = waterfall.data * maskbool

        time_guesses -= begin_time
        time_guesses = list(time_guesses)

        prof = np.mean(waterfall, axis=0)
        spec = np.mean(waterfall, axis=1)
        timebins = np.arange(0, len(prof), 1)
        freqbins = np.arange(0, len(spec), 1)

        t_width_guess = 2e-3 / (tsamp * tavg)
        print(time_guesses)
        print(amp_guesses)
        print(pulses)
        guess = [time_guesses, amp_guesses, int(512 / 2.), int(512 / 4.), t_width_guess]
        while True:
            model = fitter.gen_Gauss2D_model(
                guess[0], guess[1], f0=guess[2], bw=guess[3], dt=guess[4])
            bestfit, fitLM = fitter.fit_Gauss2D_model(waterfall, timebins, freqbins, model)
            bestfit_params, bestfit_errors = fitter.report_Gauss_parameters(bestfit, fitLM,
                                                                            verbose=True)
            fitter.plot_burst_windows(timebins, freqbins, waterfall, bestfit, ncontour=8,
                                      res_plot=True)  # diagnostic plots

            answer = input("Are you happy with the fit y/n?")
            if answer == 'y':
                break
            if answer == 'n':
                secondanswer = input("Give the intial guesses in the form "
                                     "t_sb1,t_sb2,...,t_width,f_peak,f_width")
                guessvals = [int(x.strip()) for x in secondanswer.split(',')]
                guess = [[guessvals[j] for j in range(len(current_burst_id))],
                         amp_guesses,
                         guessvals[len(current_burst_id) + 1],
                         guessvals[int(len(current_burst_id) + 2)],
                         guessvals[len(current_burst_id)]]
            else:
                print("Please provide an answer y or n.")

        os.chdir('..')

        for j in range(len(time_guesses)):
            amplitudes.append(bestfit_params[j, 0])
            t_cent.append(bestfit_params[j, 1] * tsamp * tavg)
            t_cent_e.append(bestfit_errors[j, 1] * tsamp * tavg)
            t_width.append(np.abs(bestfit_params[j, 3]) * tavg * tsamp)
            t_width_e.append(bestfit_errors[j, 3] * tsamp * tavg)
            f_cent.append(bestfit_params[j, 2] * 1.5625)
            f_cent_e.append(bestfit_errors[j, 2] * 1.5625)
            f_width.append(np.abs(bestfit_params[j, 4]) * 1.5625)
            f_width_e.append(bestfit_errors[j, 4] * 1.5625)
            gaus_angle.append(bestfit_params[j, 5])

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')
    fit_numbers = {
        'gaus amp': amplitudes,
        't_cent [s]': t_cent,
        't_cent_e [s]': t_cent_e,
        't_width [s]': t_width,
        't_width_e [s]': t_width_e,
        'f_cent [MHz]': f_cent,
        'f_cent_e [MHz]': f_cent_e,
        'f_width [MHz]': f_width,
        'f_width_e [MHz]': f_width_e,
        'f_ref [MHz]': f_ref,
        'gaus angle': gaus_angle}

    indices = [np.array([str(x) for x in bursts]), np.array(sbs)]
    df = pd.DataFrame(fit_numbers, index=indices)

    pulses = pulses.join(df)
    pulses.to_hdf(out_hdf5_file, 'pulses')
    print(pulses)
