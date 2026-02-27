import numpy as np
import matplotlib.pyplot as plt
from pycbc.types.timeseries import TimeSeries
from pycbc.types import FrequencySeries
import gengli
import sys
import bilby
sys.path.append("../ccphen/")
import ccphen
from bilby.gw.detector import PowerSpectralDensity
import numpy.typing as npt
from scipy.signal.windows import tukey
import importlib.util

spec = importlib.util.spec_from_file_location(
    "utils_3g",  # custom module name
    "../../../3G_detector_data_analysis/utils/utils.py"
)
utils_3g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_3g)




psd = PowerSpectralDensity.from_power_spectral_density_file('ET_D_psd.txt')


def save_data(filename, outdir, detector_network):
    """
    Function to save data from an interferometer network

    Saves whitened time domain strain data as a numpy .npz file with detector names as keys.

    """

    output_strain = {}
    null_stream = np.zeros(len(detector_network[0].time_array))

    for ifo in detector_network:
        whitened_data = np.array(ifo.whitened_time_domain_strain)
        null_stream += whitened_data
        output_strain[ifo.name] = whitened_data

    output_strain["null_stream"] = null_stream/np.sqrt(3)
    np.savez(f"./{outdir}/{filename}_frames.npz", **output_strain)

    return None



def generate_supernova_signal(
    target_snr: float,
    sampling_frequency: int = 4096,
    ra: float = 0.0,
    dec: float = 0.0,
    duration: float = 2.0,
    luminosity_distance: float = 1e3,
    seed_array: tuple = (7688, 763, 1263, 973, 9872),
) -> dict[str, np.ndarray]:
    """
    Generate a phenomenological core-collapse supernova gravitational wave signal.

    Uses the ccphen library to create a simulated supernova waveform with
    specified sky location and SNR, returning the frequency-domain strain
    for both polarizations.

    Parameters
    ----------
    target_snr : float
        The desired signal-to-noise ratio for the generated signal.
    sampling_frequency : int, optional
        The sampling frequency in Hz. Default is 4096.
    ra : float, optional
        Right ascension of the source in radians. Default is 0.0.
    dec : float, optional
        Declination of the source in radians. Default is 0.0.
    duration : float, optional
        Duration of the signal segment in seconds. Default is 2.0.
    luminosity_distance : float, optional
        Distance to the source in parsecs. Default is 1e3 (1 kpc).
    seed_array : tuple, optional
        Random seeds for waveform generation reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:
        - 'plus': Frequency-domain plus polarization strain
        - 'cross': Frequency-domain cross polarization strain
    """
    n_samples = int(sampling_frequency * duration)

    # Source intrinsic parameters
    parameters = ccphen.param()
    parameters.Tini = 0.2
    parameters.Tend = 1.2
    parameters.Q = 10
    parameters.npw = 3
    parameters.time_pw = [0.0, 1.0, 1.5]
    parameters.f_pw = [100, 2e3, 2.5e3]
    parameters.h_pw = [1.0, 1.0, 1.0]

    log_hrms_mean = -23.0
    log_hrms_sigma = 0.4

    waveform = ccphen.hphen_pol(
        sampling_frequency,
        n_samples,
        parameters,
        seed_array,
        luminosity_distance,
        ra,
        dec,
        log_hrms_mean=log_hrms_mean,
        log_hrms_sigma=log_hrms_sigma,
    )

    hplus = np.array(np.real(waveform.h))
    hcross = np.array(np.imag(waveform.h))

    hplus = scale_snr_with_psd(hplus, target_snr=target_snr, sample_rate=sampling_frequency)
    hcross = scale_snr_with_psd(hcross, target_snr=target_snr, sample_rate=sampling_frequency)

    fft_hplus, _ = bilby.core.utils.nfft(hplus, sampling_frequency)
    fft_hcross, _ = bilby.core.utils.nfft(hcross, sampling_frequency)




    return {'plus': fft_hplus, 'cross': fft_hcross}



def scale_snr_with_psd(time_domain_strain: npt.ArrayLike,
              target_snr: float,
              sample_rate: int = 4096,
              power_spectral_density: PowerSpectralDensity = None) -> np.ndarray:
    """
    Scale a time-domain gravitational wave strain to achieve a target SNR.

    Computes the current SNR by whitening the input strain using the provided
    power spectral density, then scales the original strain to match the
    desired SNR.

    Parameters
    ----------
    time_domain_strain : array_like
        The input time-domain strain signal to be scaled.
    target_snr : float
        The desired signal-to-noise ratio for the output strain.
    sample_rate : int, optional
        The sampling frequency in Hz. Default is 4096.
    power_spectral_density : PowerSpectralDensity, optional
        The PSD used for whitening. If None, uses the module-level ET-D PSD.

    Returns
    -------
    np.ndarray
        The scaled time-domain strain with the target SNR.
    """
    if power_spectral_density is None:
        power_spectral_density = psd

    ts = TimeSeries(time_domain_strain, sample_rate=sample_rate)
    whitened_strain = utils_3g.generate_whitened_timeseries_from_coloured_timeseries(
        ts, power_spectral_density=power_spectral_density
    )

    current_snr = np.linalg.norm(whitened_strain)
    scaling_factor = target_snr / current_snr

    return time_domain_strain * scaling_factor


def whitened_timeseries_to_coloured_timeseries(input_timeseries : TimeSeries, sampling_frequency: float, power_spectral_density: PowerSpectralDensity = psd):
    """
    Converts whitened timeseries to coloured time series

    1. Convert pycbc timeseries to pycbc frequencyseries. Call it input_frequencyseries
    2. Interpolate the power_spectral_density (N, 2) array on the sample frequency of the of input_frequencyseries
    3. Do np.sqrt(interpolated_psd) * input_frequencyseries * duration * 2. Call it coloured_frequencyseries
    4. Convert coloured pycbc frequencyseries to the coloured pycbc timeseries
    5. Return coloured pycbc timeseries

    """

    duration = len(input_timeseries) / sampling_frequency

    # Step 1: Convert to frequency series
    input_frequencyseries = input_timeseries.to_frequencyseries()

    # Step 2: Interpolate PSD at the frequencyseries sample frequencies
    freqs = np.array(input_frequencyseries.sample_frequencies)
    interpolated_psd = psd.get_power_spectral_density_array(freqs)
    interpolated_psd = np.nan_to_num(interpolated_psd, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 3: Colour the frequency series
    coloured_array = np.sqrt(interpolated_psd) * np.array(input_frequencyseries) * duration /2
    coloured_frequencyseries = FrequencySeries(
        coloured_array,
        delta_f=input_frequencyseries.delta_f,
        epoch=input_frequencyseries.epoch,
    )

    # Step 4: Convert back to time series
    coloured_pycbc_timeseries = coloured_frequencyseries.to_timeseries()

    return coloured_pycbc_timeseries


def adjust_snr(whitened_timeseries, target_snr):
    norm = np.sum(whitened_timeseries*whitened_timeseries)**0.5
    scaled_timeseries = whitened_timeseries * target_snr / norm
    return scaled_timeseries

def inject_glitch(noise_dict, n_glitches: int, seed: int, outdir: str, label: str, sampling_frequency = 4096):
    """
    1. Randmoly choose n_glitches values of times from the sample times of an item in the noise dict. 

    """

    generator = gengli.glitch_generator('L1')
    glitch_bank = generator.get_glitch(n_glitches = n_glitches, seed=seed, srate=sampling_frequency, snr = 1)
    if n_glitches ==1:
        glitch_bank = [glitch_bank]

    sample_times = np.array(list(noise_dict.values())[0].sample_times)
    glitch_injection_time = np.random.choice(sample_times, n_glitches, replace=True)

    det_names = list(noise_dict.keys())
    glitchy_interferometer = np.random.choice(len(noise_dict), n_glitches, replace=True)

    for ii in range(n_glitches):
        det_name = det_names[glitchy_interferometer[ii]]

        target_snr = np.random.uniform(0, 100, 1)[0]
        g = adjust_snr(glitch_bank[ii], target_snr)
        g = TimeSeries(g, delta_t = 1/sampling_frequency, epoch = 0.0)


        g_coloured = whitened_timeseries_to_coloured_timeseries(g, sampling_frequency=sampling_frequency)

        t = glitch_injection_time[ii]
        plot_glitches = False
        if plot_glitches:
            fig, ax = plt.subplots()
            ax.plot(g_coloured.sample_times, g_coloured)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Strain")
            ax.set_title(f"Glitch {ii} injected into {det_name} at t={t:.2f} s")
            fig.savefig(f"{outdir}/{label}_glitch_{ii}_{det_name}.pdf")
            plt.close(fig)

        g_coloured.start_time = t

        noise_dict[det_name] = noise_dict[det_name].inject(g_coloured)


    return noise_dict, glitch_injection_time, glitchy_interferometer
