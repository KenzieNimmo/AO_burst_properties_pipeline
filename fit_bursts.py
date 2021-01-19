"""
Kenzie Nimmo 2020
"""
import sys
import numpy as np
# sys.path.append('/home/nimmo/AO_burst_properties_pipeline/')
#sys.path.append('/home/jjahns/Bursts/AO_burst_properties_pipeline/')
import os
import optparse
import pandas as pd
import matplotlib.pyplot as plt
import import_fil_fits
import burst_2d_Gaussian as fitter

from scipy.stats import chisquare
from presto import psrfits


def ds(array, factor=2, axis=0):
    """Downsample using the mean along a given axis"""
    if axis < 0:
        axis += array.ndim

    axis_len = array.shape[axis]
    if axis_len % factor != 0:
        print(f"The array axis with length {axis_len} is trimmed to fit the downsample factor "
              f"{factor}.")
        slc = [slice(None)] * array.ndim
        slc[axis] = slice(-(axis_len % factor))
        array = array[tuple(slc)]

    new_shape = list(array.shape)
    new_shape[axis] //= factor
    new_shape.insert(axis+1, factor)
    array = array.reshape(new_shape).mean(axis=axis+1)
    return array


if __name__ == '__main__':
    parser = optparse.OptionParser(usage='%prog [options] infile', description="2D Gaussian fit "
                                   "to FRB data. Input the pickle file output from RFI_masker.py")
    parser.add_option('-t', '--tavg', dest='tavg', type='int', default=1,
                      help="If -t option is used, time averaging is applied using the factor "
                           "given after -t.")
    parser.add_option('-s', '--subb', type='int', default=1,
                      help="If -s option is used, subbanding is applied using the factor "
                           "given after -s.")
    parser.add_option('-p', '--pulse', type='str', default=None,
                      help="Give a pulse id to process only this pulse.")
    parser.add_option('-f', '--tfit', dest='tfit', type='float', default=5,
                      help="Give the time before and after the burst in ms that should be used "
                      "for fitting.")
    parser.add_option('--ptavg', type='int', default=1,
                      help="As -t but only used for plotting.")
    parser.add_option('--psubb', type='int', default=1,
                      help="As -s but only used for plotting.")
    parser.add_option('-g', '--pguess', action='store_true',
                      help="Plot the guessed functions")

    options, args = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    elif len(args) != 1:
        sys.stderr.write("Only one input file must be provided!\n")
    else:
        options.infile = args[-1]

    tavg = options.tavg
    subb = options.subb
    tfit = options.tfit
    ptavg = options.ptavg
    psubb = options.psubb
    plot_guess = options.pguess

    # getting the basename of the observation and pulse IDs
    basename = options.infile
    # hdf5 file
    orig_in_hdf5_file = f'../{basename}.hdf5'
    in_hdf5_file = f'{basename}_burst_properties.hdf5'

    out_hdf5_file = in_hdf5_file

    smooth = None  # smoothing value used for bandpass calibration

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')

    # Get pulses to be processed.
    if options.pulse is not None:
        pulse_ids = [options.pulse]
    else:
        pulse_ids = pulses.index.get_level_values(0).unique()
        if ('2D Gaussian', 'Amp') in pulses.columns:
            not_processed = pulses.loc[(slice(None), 'sb1'), ('2D Gaussian', 'Amp')].isna()
            pulse_ids = pulse_ids[not_processed]
    print(pulses.loc[pulse_ids])

    for pulse_id in pulse_ids:
        print("2D Gaussian fit of observation %s, pulse ID %s" % (basename, pulse_id))
        base_pulse = pulse_id.split('-')[0]
        os.chdir('%s' % base_pulse)
        filename = '%s_%s.fits' % (basename, base_pulse)

        # Get some observation parameters
        fits = psrfits.PsrfitsFile(filename)
        tsamp = fits.specinfo.dt

        # Name mask and offpulse file
        maskfile = '%s_%s_mask.pkl' % (basename, base_pulse)
        offpulsefile = '%s_%s_offpulse_time.pkl' % (basename, base_pulse)

        # read in DMs and such
        dm = pulses.loc[(pulse_id, 'sb1'), ('General','DM')]
        amp_guesses = pulses.loc[pulse_id, ('Guesses', 'Amp')]
        time_guesses = pulses.loc[pulse_id, ('Guesses', 't_cent')] / tavg
        freq_peak_guess = pulses.loc[pulse_id, ('Guesses', 'f_cent')] // subb

        # Define the window to be shown.
        begin_samp = int(time_guesses.min() - 20e-3 / (tavg * tsamp))
        end_samp = int(time_guesses.max() + 20e-3 / (tavg * tsamp))

        waterfall, t_ref, _ = import_fil_fits.fits_to_np(
            filename, dm=dm, maskfile=maskfile, AO=True, smooth_val=smooth,
            hdf5=orig_in_hdf5_file, index=base_pulse, tavg=tavg)
        waterfall = import_fil_fits.bandpass_calibration(waterfall, offpulsefile, tavg=tavg,
                                                         AO=True, smooth_val=None, plot=False)

        waterfall = waterfall[:, begin_samp:end_samp]
        t_ref += begin_samp * tsamp

        fit_mask = np.zeros(waterfall.shape, dtype=np.bool)
        fit_start = int(time_guesses.min() - begin_samp - tfit*1e-3/(tavg*tsamp))
        fit_end = int(time_guesses.max() - begin_samp + tfit*1e-3/(tavg*tsamp))
        fit_mask[:, fit_start:fit_end] = True
        fit_mask = (fit_mask & ~waterfall.mask).astype(np.float)

        freqs = fits.freqs #[~waterfall.mask[:,0]]

        if subb != 1:
            waterfall = ds(waterfall, factor=subb, axis=0)
            fit_mask = ds(fit_mask, factor=subb, axis=0)
            freqs = ds(freqs, factor=subb, axis=0)

        times = np.arange(waterfall.shape[1]) * tsamp * tavg * 1e3  # in ms
        start_stop = times[[fit_start, fit_end]]

        #waterfall = waterfall.data[~waterfall.mask[:,0]] # maskbool
        # For savings
        imjd, fmjd = psrfits.DATEOBS_to_MJD(fits.specinfo.date_obs)
        obs_start = imjd + fmjd
        fsamp = fits.specinfo.df
        f_ref = fits.specinfo.hi_freq

        time_guesses -= begin_samp
        time_guesses *= tavg*tsamp*1e3

        #freq_peak_guess = freqs[freq_peak_guess]  # n_sbs * [int(512 / subb / 2.)]
        n_sbs = pulses.loc[pulse_id].shape[0]
        freq_std_guess = [100.] * n_sbs  # n_sbs * [int(512 / subb / 4.)]
        t_std_guess = [1] * n_sbs
        amp_guesses = np.sqrt(tavg*subb) * amp_guesses.to_numpy() #/ np.sqrt((~waterfall.mask[:,0]).sum())

        # Load guesses from hdf5 file if they are saved there and use those instead
        if ('Guesses', 't_std') in pulses.columns:
            if not pulses.loc[pulse_id, ('Guesses', 't_std')].isna().loc['sb1']:
                t_std_guess = pulses.loc[pulse_id, ('Guesses', 't_std')].to_numpy()
                freq_std_guess = pulses.loc[pulse_id, ('Guesses', 'f_std')].to_numpy()

        use_standard_2D_gaussian = True
        fix_angle = False
        while True:
            if use_standard_2D_gaussian:
                model = fitter.gen_Gauss2D_model(time_guesses, amp_guesses, f0=freq_peak_guess,
                                                 bw=freq_std_guess, dt=t_std_guess)
                if fix_angle:
                    angles = model.param_names[5::6]
                    for a in angles:
                        model.fixed[a] = True
            else:
                # Use the custom drifting Gaussian model.
                model = fitter.drifting_2DGaussian(
                    amplitude=amp_guesses[0], t_mean=time_guesses[0], f_mean=freq_peak_guess[0],
                    t_stddev=t_std_guess[0], f_stddev=freq_std_guess[0], drift=0.)
                for a, tg, fg, tsg, fsg in zip(amp_guesses[1:], time_guesses[1:],
                                               freq_peak_guess[1:], t_std_guess[1:],
                                               freq_std_guess[1:]):
                    model += fitter.drifting_2DGaussian(amplitude=a, t_mean=tg, f_mean=fg,
                                                        t_stddev=tsg, f_stddev=fsg, drift=0.)
                if fix_angle:
                    angles = model.param_names[5::6]
                    for a in angles:
                        model.fixed[a] = True

            # Plot the guess
            if plot_guess:
                low_res_waterfaller = ds(ds(waterfall, psubb), ptavg, axis=1)
                fitter.plot_burst_windows(ds(times, ptavg), ds(freqs, psubb), low_res_waterfaller,
                                          model, ncontour=8, res_plot=True, vlines=start_stop)

            bestfit, fitLM = fitter.fit_Gauss2D_model(waterfall.data, times, freqs, model,
                                                      weights=fit_mask)
            bestfit_params, bestfit_errors, corr_fig = fitter.report_Gauss_parameters(bestfit,
                                                                                      fitLM,
                                                                                      verbose=True)

            timesh, freqs_m = np.meshgrid(times, freqs)
            fit_mask = fit_mask.astype(np.bool)
            timesh, freqs_m = timesh[fit_mask], freqs_m[fit_mask]
            chisq, pvalue = chisquare(waterfall[fit_mask], f_exp=bestfit(timesh, freqs_m),
                                      ddof=6*n_sbs, axis=None)
            print("Chi^2 and p-value:", chisq, pvalue)
            # Plot downsampled and subbanded as given in options
            low_res_waterfaller = ds(ds(waterfall, psubb), ptavg, axis=1)
            fig, res_fig = fitter.plot_burst_windows(ds(times, ptavg), ds(freqs, psubb),
                                                     low_res_waterfaller, bestfit, ncontour=8,
                                                     res_plot=True, vlines=start_stop)  # diagnostic plots
            if use_standard_2D_gaussian:
                corr_fig.savefig(f'{basename}_{pulse_id}_correlation')
                fig.savefig(f'{basename}_{pulse_id}_fit')
                res_fig.savefig(f'{basename}_{pulse_id}_fit_residuals')
            plt.show()

            answer = input("Are you happy with the fit y/n/skip/fix? ")
            if (answer == 'y') and use_standard_2D_gaussian:
                use_standard_2D_gaussian = False

                # Save fit parameters
                pulses.loc[pulse_id, ('2D Gaussian', 'Amp')] = bestfit_params[:, 0]
                pulses.loc[pulse_id, ('2D Gaussian', 'Amp e')] = bestfit_errors[:, 0]
                pulses.loc[pulse_id, ('2D Gaussian', 't_cent / s')] = (bestfit_params[:, 1] / 1e3
                                                                       + t_ref)
                pulses.loc[pulse_id, ('2D Gaussian', 't_cent_e / s')] = bestfit_errors[:, 1] / 1e3
                pulses.loc[pulse_id, ('2D Gaussian', 't_std / ms')] = bestfit_params[:, 3]
                pulses.loc[pulse_id, ('2D Gaussian', 't_std_e / ms')] = bestfit_errors[:, 3]
                pulses.loc[pulse_id, ('2D Gaussian', 'f_cent / MHz')] = (
                    f_ref - bestfit_params[:, 2] * fsamp * subb)
                pulses.loc[pulse_id, ('2D Gaussian', 'f_cent_e / MHz')] = (bestfit_errors[:, 2]
                                                                           * fsamp * subb)
                pulses.loc[pulse_id, ('2D Gaussian', 'f_std / MHz')] = (bestfit_params[:, 4]
                                                                        * fsamp * subb)
                pulses.loc[pulse_id, ('2D Gaussian', 'f_std_e / MHz')] = (bestfit_errors[:, 4]
                                                                          * fsamp * subb)
                pulses.loc[pulse_id, ('2D Gaussian', 'Angle')] = bestfit_params[:, 5]
                pulses.loc[pulse_id, ('2D Gaussian', 'Angle e')] = bestfit_errors[:, 5]
                pulses.loc[pulse_id, ('2D Gaussian', 'Chi^2')] = chisq
                pulses.loc[pulse_id, ('2D Gaussian', 'p-value')] = pvalue

            elif (answer == 'y') or (answer == 'skip'):
                break
            elif answer == 'fix':
                fix_angle = not fix_angle
                if fix_angle:
                    print("The angle is fixed now.")
                else:
                    print("The angle is a fittable parameter again.")
            elif answer == 'n':
                guessvals = []
                while len(guessvals) not in [n_sbs*i for i in range(1, 5)]:
                    current_gusses = np.concatenate((t_std_guess, freq_std_guess, amp_guesses,
                                                     freq_peak_guess))
                    current_gusses = str(list(current_gusses))[1:-1].replace(' ', '')
                    secondanswer = input("Give the intial guesses in ms and MHz in the"
                                         "form t_std_sb1,t_std_sb2,...,[f_std_sb1,"
                                         "...[amp_sb1,amp_sb2,...,[f_peak_sb1,...,[t_peak_sb1,...]]]] "
                                         f"with a multiple of {n_sbs} guesses in "
                                         f"total. The current values are {current_gusses}: ")
                    try:
                        guessvals = [float(x.strip()) for x in secondanswer.split(',')]
                    except:
                        print("Wrong format")

                # Save guesses.
                guessvals = np.array(guessvals)
                t_std_guess = guessvals[0 : n_sbs]
                pulses.loc[pulse_id, ('Guesses', 't_std')] = t_std_guess
                if len(guessvals) > n_sbs:
                    freq_std_guess = guessvals[n_sbs : 2*n_sbs]
                    pulses.loc[pulse_id, ('Guesses', 'f_std')] = freq_std_guess
                    if len(guessvals) > 2*n_sbs:
                        amp_guesses = guessvals[2*n_sbs:3*n_sbs]
                        pulses.loc[pulse_id, ('Guesses', 'Amp')] = (np.array(amp_guesses)
                                                                    / np.sqrt(tavg*subb))
                        if len(guessvals) > 3*n_sbs:
                            freq_peak_guess = guessvals[3*n_sbs:4*n_sbs]
                            pulses.loc[pulse_id, ('Guesses', 'f_cent')] = freq_peak_guess
# =============================================================================
#                             if len(guessvals) > 4*n_sbs:
#                                 time_guesses = guessvals[4*n_sbs:5*n_sbs]
#                                 pulses.loc[pulse_id, ('Guesses', 't_cent')] = time_guesses / tavg
# =============================================================================
            else:
                print("Please provide an answer y or n.")

        os.chdir('..')

        if answer == 'skip':
            # Save the guesses anyway
            pulses.to_hdf(out_hdf5_file, 'pulses')
            continue

        pulses.loc[pulse_id, ('Drifting Gaussian', 'Amp')] = bestfit_params[:, 0]
        pulses.loc[pulse_id, ('Drifting Gaussian', 'Amp e')] = bestfit_errors[:, 0]
        pulses.loc[pulse_id, ('Drifting Gaussian', 't_cent / s')] = (
            bestfit_params[:, 1] / 1e3 + t_ref)
        pulses.loc[pulse_id, ('Drifting Gaussian', 't_cent_e / s')] = bestfit_errors[:, 1] / 1e3
        pulses.loc[pulse_id, ('Drifting Gaussian', 't_std / ms')] = bestfit_params[:, 3]
        pulses.loc[pulse_id, ('Drifting Gaussian', 't_std_e / ms')] = bestfit_errors[:, 3]
        pulses.loc[pulse_id, ('Drifting Gaussian', 'f_cent / MHz')] = (
            f_ref - bestfit_params[:, 2] * fsamp * subb)
        pulses.loc[pulse_id, ('Drifting Gaussian', 'f_cent_e / MHz')] = (
            bestfit_errors[:, 2] * fsamp * subb)
        pulses.loc[pulse_id, ('Drifting Gaussian', 'f_std / MHz')] = (
            bestfit_params[:, 4] * fsamp * subb)
        pulses.loc[pulse_id, ('Drifting Gaussian', 'f_std_e / MHz')] = (
            bestfit_errors[:, 4] * fsamp * subb)
        pulses.loc[pulse_id, ('Drifting Gaussian', 'Drift / ms/MHz')] = bestfit_params[:, 5]
        pulses.loc[pulse_id, ('Drifting Gaussian', 'Drift e / ms/MHz')] = bestfit_errors[:, 5]
        pulses.loc[pulse_id, ('Drifting Gaussian', 'Chi^2')] = chisq
        pulses.loc[pulse_id, ('Drifting Gaussian', 'p-value')] = pvalue
        pulses.loc[pulse_id, ('General', 't_obs / MJD')] = obs_start
        pulses.loc[pulse_id, ('General', 'f_ref / MHz')] = f_ref
        pulses.loc[pulse_id, ('General', 'downsampling')] = tavg
        pulses.loc[pulse_id, ('General', 'subbanding')] = subb

        pulses.to_hdf(out_hdf5_file, 'pulses')
        print(pulses)
