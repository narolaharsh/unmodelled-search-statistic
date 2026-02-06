import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
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

# Now use it like:
# utils_3g.some_function()




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
        #print(f"{ifo.name}: shape={whitened_data.shape}, start_time={ifo.start_time}")
    output_strain["null_stream"] = null_stream
    np.savez(f"./{outdir}/{filename}.npz", **output_strain)

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

    hplus = scale_snr(hplus, target_snr=target_snr, sample_rate=sampling_frequency)
    hcross = scale_snr(hcross, target_snr=target_snr, sample_rate=sampling_frequency)

    fft_hplus, _ = bilby.core.utils.nfft(hplus, sampling_frequency)
    fft_hcross, _ = bilby.core.utils.nfft(hcross, sampling_frequency)




    return {'plus': fft_hplus, 'cross': fft_hcross}



def scale_snr(time_domain_strain: npt.ArrayLike,
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



def inject_glitch(generator, time_domain_strain, sample_rate, injection_time, start_time, target_snr):

    glitch = generator.get_glitch(snr = 1, srate = sample_rate)

    pre_factor = target_snr / np.sqrt(np.sum(glitch*glitch)) 
    glitch *= pre_factor
    glitch *= tukey(len(glitch), alpha = 0.1)
    input_time_series = TimeSeries(time_domain_strain, sample_rate=sample_rate, t0 = start_time)
    output_time_series = utils_3g.inject_glitch(glitch, input_time_series, injection_time, psd)

    return np.array(output_time_series.data)