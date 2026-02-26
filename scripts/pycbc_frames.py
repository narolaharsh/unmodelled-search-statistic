"""
Script to create ET frames using PyCBC
"""

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
from pycbc.detector import add_detector_on_earth, Detector
import utils
from bilby.gw import conversion
from tqdm import tqdm


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



def add_timeseries(noise_dict, strain_dict):
    """Add signal strain to noise timeseries for each detector.

    Parameters
    ----------
    noise_dict : dict
        Mapping of detector name to pycbc.types.timeseries.TimeSeries,
        as returned by :func:`noise_generator`.
    strain_dict : dict
        Mapping of detector name to np.ndarray of signal-only strain,
        as returned by :func:`batch_signal_generator`.

    Returns
    -------
    dict mapping detector name (str) to np.ndarray
        Signal-plus-noise strain for each detector.
    """
    return {det: noise_dict[det] + strain_dict[det] for det in strain_dict}


def write_all_frames(noise_dict, strain_dict, signal_plus_noise_dict,
                     sample_times, sampling_frequency, outdir, label, frame_duration):
    """Write noise-only, signal-only, and signal-plus-noise GWF frames for all detectors.

    Parameters
    ----------
    noise_dict : dict
        Mapping of detector name to pycbc.types.timeseries.TimeSeries.
    strain_dict : dict
        Mapping of detector name to np.ndarray of signal-only strain.
    signal_plus_noise_dict : dict
        Mapping of detector name to np.ndarray of signal-plus-noise strain.
    sample_times : np.ndarray
        Sample times of the strain arrays; ``sample_times[0]`` is used as the epoch.
    sampling_frequency : float
        Sampling frequency in Hz.
    outdir : str
        Output directory (must already exist).
    label : str
        Label used as a prefix for all output filenames.
    frame_duration : float
        Frame duration in seconds, used in the noise filename for bookkeeping.
    """
    epoch = sample_times[0]
    delta_t = 1 / sampling_frequency

    for det, ts_noise in noise_dict.items():
        channel = f"{det}:STRAIN"
        write_frame(f"{outdir}/{label}_noise_only_{det}_duration_{frame_duration}.gwf",
                    channel, ts_noise)

    for det, strain in strain_dict.items():
        channel = f"{det}:STRAIN"
        ts = TimeSeries(strain, delta_t=delta_t, epoch=epoch)
        write_frame(f"{outdir}/{label}_{det}_signal_only.gwf", channel, ts)

    for det, spn in signal_plus_noise_dict.items():
        channel = f"{det}:STRAIN"
        ts = TimeSeries(spn, delta_t=delta_t, epoch=epoch)
        write_frame(f"{outdir}/{label}_{det}_signal_and_noise.gwf", channel, ts)


def noise_generator(detector_network, sampling_frequency, frame_duration,
                    minimum_frequency, seed):
    """Generate coloured noise for each detector in the network.

    Parameters
    ----------
    detector_network : str
        ``'ETT'`` for the triangle configuration or ``'ET2L'`` for the 2L design.
    sampling_frequency : float
        Sampling frequency in Hz.
    frame_duration : float
        Duration of the noise frame in seconds.
    minimum_frequency : float
        Lower frequency cutoff in Hz.
    seed : int
        Base random seed; each detector uses ``seed + i`` to stay reproducible.

    Returns
    -------
    dict mapping detector name (str) to pycbc.types.timeseries.TimeSeries.
    """
    delta_f = 1.0 / 8
    length = int(sampling_frequency / 2 / delta_f) + 1
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))

    if detector_network == "ETT":
        psd_file = './noise_curves/ET_D_psd.txt'
        detectors = ["E1", "E2", "E3"]
    elif detector_network == "ET2L":
        psd_file = './noise_curves/ET_D_psd_15km.txt'
        detectors = ["ETLim", "ETSar"]
    else:
        raise ValueError("Detector network does not exist")

    with chdir(_scripts_dir):
        power_spectral_density = pycbc.psd.from_txt(
            os.path.join(_scripts_dir, psd_file), length, delta_f,
            minimum_frequency, is_asd_file=False)

    noise_dict = {}
    for i, det in enumerate(detectors):
        detector_noise = colored_noise(psd=power_spectral_density, start_time=0,
                                       end_time=frame_duration, seed=seed + i,
                                       sample_rate=sampling_frequency,
                                       low_frequency_cutoff=minimum_frequency,
                                       filter_duration=128)
        epoch = lal.LIGOTimeGPS(detector_noise.sample_times[0])
        noise_dict[det] = TimeSeries(np.array(detector_noise),
                                           delta_t=1. / sampling_frequency, epoch=epoch)

    return noise_dict



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


_NETWORK_CONFIG = {
    'ET2L': (['ETLim', 'ETSar'], [Detector('ETLim'), Detector('ETSar')]),
    'ETT':  (['E1', 'E2', 'E3'],  [Detector('E1'),   Detector('E2'),   Detector('E3')]),
}


def batch_signal_generator(injection_catalog, injection_times, detector_network,
                            sample_times, sampling_frequency, minimum_frequency,
                            reference_frequency, approximant='IMRPhenomTPHM'):
    """Build the network, initialise strain arrays, and inject all signals.

    Parameters
    ----------
    injection_catalog : dict
        Dictionary with keys ``'injection_0'``, ``'injection_1'``, … containing
        source parameter dicts (as loaded from the JSON catalog).
    injection_times : array-like
        Geocentric coalescence times, one per injection.
    detector_network : str
        ``'ETT'`` or ``'ET2L'``; used to select the detector objects and
        initialise zero-valued strain arrays of the correct length.
    sample_times : np.ndarray
        Sample times of the strain arrays (defines the time window).
    sampling_frequency : float
        Sampling frequency in Hz.
    minimum_frequency : float
        Lower frequency cutoff for waveform generation in Hz.
    reference_frequency : float
        Reference frequency for spin definitions in Hz.
    approximant : str, optional
        PyCBC waveform approximant. Default is ``'IMRPhenomTPHM'``.

    Returns
    -------
    dict mapping detector name (str) to np.ndarray
        Per-detector strain arrays with all signals injected.
    """
    if detector_network not in _NETWORK_CONFIG:
        raise NotImplementedError(f"No such detector network: '{detector_network}'")

    det_names, network = _NETWORK_CONFIG[detector_network]
    n_samples = len(sample_times)
    strain_dict = {name: np.zeros(n_samples) for name in det_names}

    for ii in tqdm(range(len(injection_times))):
        parameters = convert_parameters(injection_catalog[f"injection_{ii}"])
        parameters['geocent_time'] = injection_times[ii]

        detector_frame_signal_list = signal_generator(parameters, network, approximant,
                                                      sampling_frequency, minimum_frequency,
                                                      reference_frequency, earth_rotation=True)

        for name, signal in zip(det_names, detector_frame_signal_list):
            strain_dict[name] = inject_signal_into_strain(strain_dict[name], signal,
                                                          sample_times, sampling_frequency)
    return strain_dict


def plot_timeseries(noise_dict, signal_dict, sample_times, outdir, label):
    """Plot noise-only and signal strain timeseries to PDF files.

    Produces two figures:
    - ``<label>_noise.pdf``: one panel per detector showing noise-only strain.
    - ``<label>_strain.pdf``: one panel per detector showing signal-only strain,
      plus a final panel with the null stream (sum over detectors).

    Parameters
    ----------
    noise_dict : dict
        Mapping of detector name to pycbc.types.timeseries.TimeSeries.
    signal_dict : dict
        Mapping of detector name to np.ndarray of signal-only strain,
        as returned by :func:`batch_signal_generator`.
    sample_times : np.ndarray
        Sample times corresponding to the signal strain arrays.
    outdir : str
        Output directory for the saved figures.
    label : str
        Label used as a prefix for the output filenames.
    """
    n_det = len(noise_dict)

    fig, axes = plt.subplots(n_det, 1, figsize=(10, 3 * n_det), sharex=True)
    for ax, (det, ts) in zip(axes, noise_dict.items()):
        ax.plot(ts.sample_times, ts)
        ax.set_ylabel(f"{det} Strain")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{label}_noise.pdf")
    plt.close(fig)

    strain_array = np.array(list(signal_dict.values()))
    null_stream = np.sum(strain_array, axis=0)

    fig, axes = plt.subplots(n_det + 1, 1, figsize=(10, 3 * n_det), sharex=True, sharey=True)
    for ax, strain in zip(axes, strain_array):
        ax.plot(sample_times, strain)
        ax.set_ylabel("Strain")
    axes[-1].plot(sample_times, null_stream)
    axes[-1].set_ylabel("Null stream")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    fig.savefig(f"{outdir}/{label}_strain.pdf")
    plt.close(fig)


def inject_glitch():
    return 

def main():

    args = parse_args()
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    np.random.seed(args.seed)

    injection_catalog = json.load(open(args.signal_catalog, "r"))
    reference_frequency = 50.0

    signal_injection_times = np.random.uniform(0, args.frame_duration, args.n_signals)
    np.savetxt(f"{args.outdir}/{args.label}_injection_times.txt", signal_injection_times, fmt="%.6f")
    
    ########################################
    ###### Generate coloured noise ########
    noise_dict = noise_generator(args.detector_network, args.sampling_frequency,
                                 args.frame_duration, args.minimum_frequency, args.seed)

   
    
    ########################################
    ## Generate all signals ################
    total_time = int(args.frame_duration)
    sample_times = np.linspace(start=0, stop=total_time, num=total_time*args.sampling_frequency, endpoint=True)
    
    signal_dict = batch_signal_generator(injection_catalog, signal_injection_times,
                                         args.detector_network, sample_times,
                                         args.sampling_frequency, args.minimum_frequency,
                                         reference_frequency)


    ##############################################
    ###### Add signals to noise  ################
    signal_plus_noise_dict = add_timeseries(noise_dict, signal_dict)

    write_all_frames(noise_dict, signal_dict, signal_plus_noise_dict,
                     sample_times, args.sampling_frequency,
                     args.outdir, args.label, args.frame_duration)

    if args.plot_timeseries:
        plot_timeseries(noise_dict, signal_dict, sample_times, args.outdir, args.label)







if __name__ == "__main__":
    main()