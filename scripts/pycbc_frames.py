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
from pycbc.detector import add_detector_on_earth, Detector
import utils
from bilby.gw import conversion
from tqdm import tqdm
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


add_detector_on_earth("ETSar", longitude = 9.4167 * np.pi / 180, latitude = 40.5167 * np.pi/180, yangle=(160.5674 * np.pi / 180.0),
                      xlength=15e3, ylength=15e3)



add_detector_on_earth("ETLim", longitude = 5.92056 * np.pi/180, latitude = 50.72305 * np.pi/180, yangle=(115.0 * np.pi / 180),
                      xlength=15e3, ylength=15e3)



def noise_generator(args):
    """
    Generate coloured noise using PyCBC.
    """
    
    delta_f = 1.0/8
    length = int(args.sampling_frequency/2/delta_f) + 1
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))

    if args.detector_network == "ETT":
        psd_file = './noise_curves/ET_D_psd.txt'
        detectors = ["E1", "E2", "E3"]
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
    earth_rotation : bool

    Returns
    -------
    list of pycbc TimeSeries, one per detector.
    """
    if earth_rotation:
        return [det.project_wave(hp=hp, hc=hc, ra=ra, dec=dec, polarization=psi)
                for det in detectors]
    else:
        antenna_patterns = get_antenna_patterns(detectors, ra, dec, psi, geocent_time)
        return [fp*hp + fc*hc for fp, fc in antenna_patterns]


def signal_generator(parameters: dict, detector_network: list, approximant: str, sampling_frequency: float,
                 minimum_frequency: float, reference_frequency: float, earth_rotation: bool):
    
    """Generate detector-frame time-domain signals for a network of detectors.

    Computes the plus and cross polarizations of a CBC waveform using
    get_td_waveform, sets the coalescence time from parameters['geocent_time'],
    then projects onto each detector via project_hphc_to_detectors.

    Parameters
    ----------
    parameters : dict
        Source parameters. Required keys:
        mass_1, mass_2,
        spin1x, spin1y, spin1z,
        spin2x, spin2y, spin2z,
        distance, coa_phase, inclination,
        geocent_time, ra, dec, psi.
    detector_network : list of pycbc.detector.Detector
        Detectors onto which the signal is projected.
    approximant : str
        PyCBC waveform approximant (e.g. 'IMRPhenomTPHM').
    sampling_frequency : float
        Sampling frequency in Hz; sets delta_t = 1 / sampling_frequency.
    minimum_frequency : float
        Lower frequency cutoff for waveform generation in Hz.
    reference_frequency : float
        Reference frequency for spin definitions in Hz.
    earth_rotation : bool
        Whether to account for Earth rotation when projecting onto detectors.

    Returns
    -------
    list of pycbc.types.timeseries.TimeSeries
        Detector-frame strain time series, one per detector in detector_network.
    """
    
    hp, hc = get_td_waveform(approximant = approximant,
                             mass1 = parameters['mass_1'],
                             mass2 = parameters['mass_2'],
                             spin1x = parameters['spin_1x'], 
                             spin1y = parameters['spin_1y'],
                             spin1z = parameters['spin_1z'],
                             spin2x = parameters['spin_2x'],
                             spin2y = parameters['spin_2y'],
                             spin2z = parameters['spin_2z'],
                             distance = parameters['luminosity_distance'],
                             coa_phase = parameters['phase'],
                             inclination = parameters['theta_jn'],
                             f_lower = minimum_frequency,
                             f_ref = reference_frequency,
                             delta_t = 1/sampling_frequency)
    
    hp.start_time += parameters['geocent_time']
    hc.start_time += parameters['geocent_time']

    
    ht_list = project_hphc_to_detectors(detector_network, hp, hc, parameters['ra'], parameters['dec'], parameters['psi'], parameters['geocent_time'], earth_rotation)
    return ht_list


def convert_parameters(parameters, reference_frequency = 50.0):
    parameters['reference_frequency'] = reference_frequency
    parameters = conversion.generate_mass_parameters(parameters, source = False)

    parameters = conversion.generate_component_spins(parameters)

    return parameters




def inject_signal_into_strain(strain_series, signal, time_series, sampling_frequency):
    """Inject a detector-frame signal into a strain array, handling four overlap cases.

    Parameters
    ----------
    strain_series : np.ndarray
        The strain array to inject the signal into (modified in-place).
    signal : array-like
        The detector-frame signal (supports len() and slicing).
    time_series : array-like
        Sample times of the strain array (first and last elements define the window).
    sampling_frequency : float
        Sampling frequency in Hz.
    """
    signal_time = np.array(signal.sample_times)
    signal_start_time = signal_time[0]
    signal_end_time = signal_time[-1]

    if signal_start_time >= time_series[0] and signal_end_time <= time_series[-1]:
        # Signal fully inside the strain window
        time_index = int(signal_start_time * sampling_frequency + 0.5)
        strain_series[time_index:time_index + len(signal)] += signal

    elif signal_start_time <= time_series[0] and signal_end_time <= time_series[-1]:
        # Signal starts before the window, ends inside
        time_index = int(signal_end_time * sampling_frequency + 0.5)
        strain_series[:time_index] += signal[len(signal) - time_index:]

    elif signal_start_time >= time_series[0] and signal_end_time >= time_series[-1]:
        # Signal starts inside the window, ends after
        time_index = int(signal_start_time * sampling_frequency + 0.5)
        strain_series[time_index:] += signal[:len(strain_series) - time_index]

    else:
        # Signal fully spans the strain window
        time_index = int((time_series[0] - signal_start_time) * sampling_frequency + 0.5)
        strain_series[:] += signal[time_index:time_index + len(strain_series)]

    return strain_series



def main():

    args = parse_args()
    np.random.seed(args.seed)

    injection_catalog = json.load(open(args.signal_catalog, "r"))
    reference_freqeuncy = 50.0
    signal_injection_times = np.random.uniform(0, args.frame_duration, args.n_signals)

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    np.savetxt(f"{args.outdir}/{args.label}_injection_times.txt", signal_injection_times, fmt="%.6f")

    noise_timeseries = noise_generator(args)
    total_time = int(args.frame_duration)
    
    sample_times = np.linspace(start=0, stop=total_time, num=total_time*args.sampling_frequency, endpoint=True)

    if args.detector_network == 'ET2L':
        strain_list = [np.zeros(total_time*args.sampling_frequency), np.zeros(total_time*args.sampling_frequency)]
        network = [Detector('ETLim'), Detector('ETSar')]
    elif args.detector_network == 'ETT':
        strain_list = [np.zeros(total_time*args.sampling_frequency), np.zeros(total_time*args.sampling_frequency), np.zeros(total_time*args.sampling_frequency)]
        network = [Detector('E1'), Detector('E2'), Detector('E3')]
    else:
        raise NotImplementedError("No such detector.")
    
    ## Generate all signals ####
    for ii in tqdm(range(args.n_signals)):

        parameters = convert_parameters(injection_catalog[f"injection_{ii}"])
        parameters['geocent_time'] = signal_injection_times[ii]

        detector_frame_signal_list = signal_generator(parameters, network, 'IMRPhenomTPHM', 
                                                 args.sampling_frequency, args.minimum_frequency, reference_freqeuncy, 
                                                 earth_rotation=True)
        
        for jj in range(len(strain_list)):
            strain_list[jj] = inject_signal_into_strain(strain_list[jj], 
                                                        detector_frame_signal_list[jj], 
                                                        sample_times,
                                                        args.sampling_frequency)
    

    ## Add signals to noise ####
    for jj, det in enumerate(network):
        channel = f"{det.name}:STRAIN"

        ts_signal_only = TimeSeries(strain_list[jj], delta_t=1/args.sampling_frequency, epoch=sample_times[0])
        write_frame(f"{args.outdir}/{args.label}_{det.name}_signal_only.gwf", channel, ts_signal_only)

        signal_plus_noise = noise_timeseries[det.name] + strain_list[jj]
        ts_signal_noise = TimeSeries(signal_plus_noise, delta_t=1/args.sampling_frequency, epoch=sample_times[0])
        write_frame(f"{args.outdir}/{args.label}_{det.name}_signal_and_noise.gwf", channel, ts_signal_noise)

    strain_array = np.array(strain_list)
    null_stream = np.sum(strain_array, axis = 0)

    if args.plot_timeseries:
        n_det = len(noise_timeseries)
        fig, axes = plt.subplots(n_det, 1, figsize=(10, 3 * n_det), sharex=True)
        for ax, (det, ts) in zip(axes, noise_timeseries.items()):
            ax.plot(ts.sample_times, ts)
            ax.set_ylabel(f"{det} Strain")
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/{args.label}_noise.pdf")

        fig, axes = plt.subplots(n_det+1, 1, figsize=(10, 3 * n_det), sharex=True, sharey = True)
        for ax, strain in zip(axes, strain_array):
            ax.plot(sample_times, strain)
            ax.set_ylabel(f"Strain")
        axes[-1].plot(sample_times, null_stream)
        axes[-1].set_ylabel("Null stream")
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/{args.label}_strain.pdf")







if __name__ == "__main__":
    main()