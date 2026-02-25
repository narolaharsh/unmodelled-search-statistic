import numpy as np
import matplotlib.pyplot as plt
import pycbc
import gengli
import os
from contextlib import chdir
import argparse
import json
import logging
from pycbc.noise.reproduceable import colored_noise
from pycbc.frame import write_frame
from pycbc.types.timeseries import TimeSeries
import lal
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector


"""
Script to create ET frames using PyCBC
"""



def parse_args():
    parser = argparse.ArgumentParser(description="Generate whitened frame files in npz format for ET detector with gaussian noise and optional signals/glitches.")
    parser.add_argument("--seed", type=int, default=2323, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./deleteme", help="Output directory")
    parser.add_argument("--label", type=str, default="deleteme", help="Label for output files")
    parser.add_argument("--frame-duration", type=int, default=256, help="Frame duration in seconds")
    parser.add_argument("--sampling-frequency", type=int, default=4096, help="Sampling frequency in Hz")
    parser.add_argument("--minimum-frequency", type=float, default=20, help="Minimum frequency in Hz")
    parser.add_argument("--start-time", type=float, default=3600, help="Start time in seconds")
    parser.add_argument("--inject-glitches", type=int, default=0, help="Set to 0 if you do not want to inject glitches")
    parser.add_argument("--inject-signals", type=int, default = 1, help="0 will return Gaussian nosie, 1 will inject CBC, 2 will inject SN signals")
    parser.add_argument("--n-signals", type=int, default=1, help="Number of signals to inject")
    parser.add_argument("--n-glitches", type=int, default=1, help="Number of glitches to inject")
    parser.add_argument("--padding", type=float, default=5.0, help="Length of the segments near the edge where we do not inject anything")
    parser.add_argument("--plot-timeseries", type = int, default = 1, help = "Set to 1 if you want to make plots for sanity checks.")
    parser.add_argument("--signal-catalog", type = str, help = 'Injection file from which the parameters should be read')
    parser.add_argument("--detector-network", type = str, default = 'ETT', help = "ETT for triangle and ET2L for the 2L design")
    return parser.parse_args()



def generate_noise(args):
    """
    Generate coloured noise using PyCBC.
    """
    
    delta_f = 1.0/8
    length = int(args.sampling_frequency/2/delta_f) + 1
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))

    if args.detector_network == "ETT":
        psd_file = './noise_curves/ET_D_psd.txt'
        detectors = ["ET1", "ET2", "ET3"]
    elif args.detector_network == "ET2L":
        psd_file = './noise_curves/ET_D_psd_15km.txt'
        detectors = ["ETLim", "ETSar"]
    else:
        raise ValueError("Detector network does not exist")


    with chdir(_scripts_dir):
        power_spectral_density = pycbc.psd.from_txt(os.path.join(_scripts_dir, psd_file), length,
                                                    delta_f,
                                                    args.minimum_frequency, is_asd_file = False)

    noise_timeseries = {}
    for i, det in enumerate(detectors):
        detector_noise = colored_noise(psd=power_spectral_density, start_time=0, end_time=args.frame_duration,
                                       seed=args.seed + i, sample_rate=args.sampling_frequency,
                                       low_frequency_cutoff=args.minimum_frequency, filter_duration=128)
        epoch = lal.LIGOTimeGPS(detector_noise.sample_times[0])
        ts = TimeSeries(np.array(detector_noise), delta_t=1./args.sampling_frequency, epoch=epoch)
        write_frame(f"{args.outdir}/{args.label}_noise_only_{det}_duration_{args.frame_duration}.gwf", f"{det}:STRAIN", ts)
        noise_timeseries[det] = ts

    return noise_timeseries


def get_antenna_patterns(detectors, ra, dec, psi, geocent_time):
    """
    Compute the plus and cross antenna pattern factors for a list of detectors.

    Parameters
    ----------
    detectors : list of pycbc.detector.Detector
    ra, dec, psi : float
        Sky location and polarization angle (radians).
    geocent_time : float
        GPS time at geocenter.

    Returns
    -------
    list of (fp, fc) tuples, one per detector.
    """
    return [
        det.antenna_pattern(right_ascension=ra, declination=dec,
                            polarization=psi, t_gps=geocent_time)
        for det in detectors
    ]


def project_hphc_to_detectors(detectors, hp, hc, ra, dec, psi, geocent_time, earth_rotation):
    """
    From hp and hc, compute the detector frame signal

    Parameters
    ----------
    detectors : list of pycbc.detector.Detector
    hp, hc : pycbc TimeSeries
        Plus and cross polarisations.
    ra, dec, psi : float
        Sky location and polarization angle (radians).
    geocent_time : float
        GPS time at geocenter.
    earth_rotation : str
        "True"  – use project_wave (accounts for Earth rotation).
        "False" – use static antenna patterns (fp*hp + fc*hc).

    Returns
    -------
    list of pycbc TimeSeries, one per detector.
    """
    if earth_rotation == "True":
        return [det.project_wave(hp=hp, hc=hc, ra=ra, dec=dec, polarization=psi)
                for det in detectors]
    elif earth_rotation == "False":
        antenna_patterns = get_antenna_patterns(detectors, ra, dec, psi, geocent_time)
        return [fp*hp + fc*hc for fp, fc in antenna_patterns]
    else:
        raise NotImplementedError("Must choose from earth rotation 'True' or 'False'.")


def signal_generator(detector, approximant, sampling_frequency,
                 minimum_frequency, reference_frequency, geocent_time, parameters, earth_rotation):
    
    """
    Return detector frame signal
    """
    
    hp, hc = get_td_waveform(approximant = approximant,
                             mass_1 = parameters['mass_1'],
                             mass_2 = parameters['mass_2'],
                             spin1x = parameters['spin1x'], 
                             spin1y = parameters['spin1y'],
                             spin1z = parameters['spin1z'],
                             spin2x = parameters['spin2x'],
                             spin2y = parameters['spin2y'],
                             spin2z = parameters['spin2z'],
                             distance = parameters['distance'],
                             coa_phase = parameters['coa_phase'],
                             inclination = parameters['inclination'],
                             f_lower = minimum_frequency,
                             f_ref = reference_frequency,
                             delta_t = 1/sampling_frequency)
    
    hp.start_time += geocent_time
    hc.start_time += geocent_time

    ra = parameters['ra']
    dec = parameters['dec']
    psi = parameters['polarization']

    if detector == 'ETT':
        detector_1 = Detector("ET1")
        detector_2 = Detector("ET2")
        detector_3 = Detector("ET3")
        detector_list = [detector_1, detector_2, detector_3]
    elif detector == 'ET2L':
        detector_1 = Detector("ETLim")
        detector_2 = Detector("ETSar")
        detector_list = [detector_1, detector_2]


    else:
        raise NotImplementedError("No such detector.")
    
    ht_list = project_hphc_to_detectors(detector_list, hp, hc, ra, dec, psi, geocent_time, earth_rotation)

    return ht_list


def main():

    args = parse_args()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)


    noise_timeseries = generate_noise(args)

    if args.plot_timeseries:
        n_det = len(noise_timeseries)
        fig, axes = plt.subplots(n_det, 1, figsize=(10, 3 * n_det), sharex=True)
        for ax, (det, ts) in zip(axes, noise_timeseries.items()):
            ax.plot(ts.sample_times, ts)
            ax.set_ylabel(f"{det} Strain")
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/{args.label}.pdf")







if __name__ == "__main__":
    main()