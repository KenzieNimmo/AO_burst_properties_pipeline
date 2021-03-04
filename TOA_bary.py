###requires astropy 2.0 and python3. Will not work with numpy v1.14 due to a numpy function bug###
###Kenzie Nimmo 2020

import sys
import argparse
import pandas as pd

from math import pi
from astropy import time, coordinates as coord, units as u, constants as const


def get_bary(toas, source, location):
    times = time.Time(toas, format='mjd', scale='utc', location=location)
    ltt_bary = times.light_travel_time(source)  # time offset between barycentre and Earth
    toas_bary = times.tdb + ltt_bary
    return toas_bary.value


def barycorr(obs_start, burst_times, f_ref, dms, FRB='R1', telescope='Eff'):
    """
    obs_start is the start time of the scan in MJD.
    burst_time is the time of the burst relative to the obs_start in seconds.
    f_ref is the reference frequency to correct to infinite frequency.
    dm is the dispersion measure.
    FRB_RA is the right ascension of the FRB in the format 01:58:00.7502
    FRB_dec is the declination of the FRB in the format +65:43:00.3152
    telescope  is the telescope you used for your observation (Eff, CHIME, DSS43)
    """
    FRB = str(FRB)
    if FRB == 'R3':
        FRB_coord = coord.SkyCoord("01:58:00.7502", "+65:43:00.3152", unit=(u.hourangle, u.deg),
                                   frame='icrs')  # R3 obs pos
    if FRB == 'R1':
        FRB_coord = coord.SkyCoord("05:31:58.698", "+33:08:52.586", unit=(u.hourangle, u.deg),
                                   frame='icrs')

    telescope = str(telescope)
    if telescope == 'Eff':
        telescope_coord = coord.EarthLocation.from_geodetic(
            lon=(06. + 53./60. + 00.99425/3600.)*u.deg,  # Effelsberg geodetic coords in deg
            lat=(50. + 31./60. + 29.39459/3600.)*u.deg,
            height=369.082*u.m)
    if telescope == 'CHIME':
        telescope_coord = coord.EarthLocation.from_geodetic(
            lon=(-119. + 37./60. + 26./3600.)*u.deg,  # CHIME geodetic coords in deg
            lat=(49. + 19./60.+16./3600.)*u.deg,
            height=545.0*u.m)
    if telescope == 'DSS43':
        telescope_coord = coord.EarthLocation.from_geodetic(
            lon=(148. + 58./60. + 52.55394/3600.)*u.deg,  # DSS-43 geodetic coords in deg
            lat=(35. + 24./60. + 8.74388/3600.)*u.deg,
            height=689.608*u.m)
    if telescope == 'Arecibo':
        telescope_coord = coord.EarthLocation.from_geodetic(
            lon=-(66. + 45./60. + 11.1/3600.)*u.deg,  # arecibo geodetic coords in deg
            lat=(18. + 20./60. + 36.6/3600.)*u.deg,
            height=497.*u.m)

    # obs_start from the filterbank header. Top of the top frequency channel (readfile reports
    # mid of channel) #in MJD
    burst_time_MJD = obs_start + burst_times/(24.*3600.)

    # TOA_correctDM = (TOA-dm_shift)
    TOA_bary = get_bary(burst_time_MJD, source=FRB_coord, location=telescope_coord)

    dm_const = (const.e.gauss**2/(2*pi*const.m_e*const.c)).to(u.cm**3/u.pc*u.MHz**2*u.s)

    dm_shift = (dm_const.value*(1./(f_ref)**2)*dms)/(24.*3600.)
    return TOA_bary, TOA_bary - dm_shift


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        usage='%prog infile_basename',
        description="Correcting the peak time of the burst to the Solar System Barycentre, "
        "corrected to infinite frequency")
    parser.add_argument('infile_basename')
    args = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    elif len(sys.argv) != 2:
        sys.stderr.write("Only one input file must be provided!\n")
    else:
        basename = args.infile_basename

    in_hdf5_file = '%s_burst_properties.hdf5' % basename

    out_hdf5_file = in_hdf5_file

    pulses = pd.read_hdf(in_hdf5_file, 'pulses')

    #pulse_ids = pulses.index.get_level_values(0).unique()
    #if 'TOA / MJD' in pulses.columns:
    #    not_processed = pulses.loc[(slice(None), 'sb1'), 'TOA / MJD'].isna()
    #    pulse_ids = pulse_ids[not_processed]

    print(f"Finding the time of arrivals (barycentre corrected) of observation {basename}")

    # beginning MJD of file
    obs_start = pulses[('General', 't_obs / MJD')]  # MJD
    # dm of burst
    dm = pulses[('General','DM')]
    # Reference Frequency, using the top of the band as reference (same as PRESTO and
    # spectra.dedisperse from psrfits.py)
    ref_freq = pulses[('General', 'f_ref / MHz')]

    for model_name in ['2D Gaussian','Drifting Gaussian']:
        # time of pulse in file
        t_burst = pulses[(model_name, 't_cent / s')]  # in seconds

        toas_ref_freq, toas = barycorr(obs_start, t_burst, ref_freq, dm, FRB='R1', telescope='Arecibo')

        pulses[(model_name, 'TOA at f_ref / MJD')] = toas_ref_freq
        pulses[(model_name, 'TOA / MJD')] = toas

    pulses.to_hdf(out_hdf5_file, 'pulses')
