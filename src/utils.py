import numpy as np
import matplotlib.pyplot as plt
from pycbc.types.timeseries import TimeSeries
import gengli
# sys.path.append("../ccphen/")
from pycbc.filter import sigma as pycbc_sigma
import pycbc
# import importlib.util

DEBUG = False


def whitened_timeseries_to_coloured_timeseries(
        input_timeseries: TimeSeries, sampling_frequency: float, power_spectral_density):
    """
    Converts whitened timeseries to coloured time series

    1. Convert pycbc timeseries to pycbc frequencyseries. Call it input_frequencyseries
    2. Interpolate the power_spectral_density (N, 2) array on the sample frequency of the of input_frequencyseries
    3. Do np.sqrt(interpolated_psd) * input_frequencyseries * duration * 2. Call it coloured_frequencyseries
    4. Convert coloured pycbc frequencyseries to the coloured pycbc timeseries
    5. Return coloured pycbc timeseries

    """
    # Step 1: Convert to frequency series
    input_frequencyseries = input_timeseries.to_frequencyseries()

    # Step 2: Interpolate PSD at the frequencyseries sample frequencies
    interpolated_psd = pycbc.psd.interpolate(power_spectral_density, input_timeseries.delta_f)
    interpolated_psd.data[~np.isfinite(interpolated_psd.data)] = 0.0
    interpolated_asd = interpolated_psd ** 0.5
    coloured_frequencyseries = (interpolated_asd) * input_frequencyseries
    coloured_pycbc_timeseries = coloured_frequencyseries.to_timeseries()

    return coloured_pycbc_timeseries


def adjust_snr(coloured_time_series, target_snr, minimum_frequency, power_spectral_density):

    psd_interp = pycbc.psd.interpolate(power_spectral_density, coloured_time_series.delta_f)
    current_sigma_square = float(pycbc_sigma(coloured_time_series, psd=psd_interp,
                                             low_frequency_cutoff=minimum_frequency))
    current_snr = current_sigma_square ** 0.5

    coloured_time_series = coloured_time_series * target_snr / current_snr
    return coloured_time_series


def inject_glitch(noise_dict, n_glitches: int, minimum_frequency: float,
                  power_spectral_density, seed: int, outdir: str, label: str,
                  sampling_frequency=4096):
    """
    1. Randomly choose n_glitches values of times from the sample times of an item in the noise dict.

    """
    generator = gengli.glitch_generator('L1')
    glitch_bank = generator.get_glitch(n_glitches=n_glitches, seed=seed, srate=sampling_frequency, snr=1)
    if n_glitches == 1:
        glitch_bank = [glitch_bank]

    sample_times = np.array(list(noise_dict.values())[0].sample_times)
    glitch_injection_time = np.random.choice(sample_times, n_glitches, replace=True)
    glitch_snrs = np.random.uniform(0, 200, n_glitches)
    det_names = list(noise_dict.keys())
    glitchy_interferometer = np.random.choice(len(noise_dict), n_glitches, replace=True)

    for ii in range(n_glitches):
        det_name = det_names[glitchy_interferometer[ii]]
        g = TimeSeries(glitch_bank[ii], delta_t=1 / sampling_frequency, epoch=0.0)

        g_coloured = whitened_timeseries_to_coloured_timeseries(
            g, power_spectral_density=power_spectral_density, sampling_frequency=sampling_frequency)
        g_coloured = adjust_snr(g_coloured, glitch_snrs[ii], minimum_frequency, power_spectral_density)

        t = glitch_injection_time[ii]

        if DEBUG:
            fig, ax = plt.subplots()
            ax.plot(g_coloured.sample_times, g_coloured)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Strain")
            ax.set_title(f"Glitch {ii} injected into {det_name} at t={t:.2f} s")
            fig.savefig(f"{outdir}/{label}_glitch_{ii}_{det_name}.pdf")
            plt.close(fig)

        g_coloured.start_time = t

        noise_dict[det_name] = noise_dict[det_name].inject(g_coloured)

    return noise_dict, glitch_injection_time, glitchy_interferometer, glitch_snrs


# Under construction
# def generate_supernova_signal(
#     target_snr: float,
#     sampling_frequency: int = 4096,
#     ra: float = 0.0,
#     dec: float = 0.0,
#     duration: float = 2.0,
#     luminosity_distance: float = 1e3,
#     seed_array: tuple = (7688, 763, 1263, 973, 9872),
# ) -> dict[str, np.ndarray]:
#     """
#     Generate a phenomenological core-collapse supernova gravitational wave signal.
#     """
#     n_samples = int(sampling_frequency * duration)
#     parameters = ccphen.param()
#     parameters.Tini = 0.2
#     parameters.Tend = 1.2
#     parameters.Q = 10
#     parameters.npw = 3
#     parameters.time_pw = [0.0, 1.0, 1.5]
#     parameters.f_pw = [100, 2e3, 2.5e3]
#     parameters.h_pw = [1.0, 1.0, 1.0]
#     log_hrms_mean = -23.0
#     log_hrss_sigma = 0.4
#     waveform = ccphen.hphen_pol(
#         sampling_frequency, n_samples, parameters, seed_array,
#         luminosity_distance, ra, dec,
#         log_hrms_mean=log_hrms_mean, log_hrms_sigma=log_hrms_sigma,
#     )
#     hplus = np.array(np.real(waveform.h))
#     hcross = np.array(np.imag(waveform.h))
#     hplus = scale_snr_with_psd(hplus, target_snr=target_snr, sample_rate=sampling_frequency)
#     hcross = scale_snr_with_psd(hcross, target_snr=target_snr, sample_rate=sampling_frequency)
#     fft_hplus, _ = bilby.core.utils.nfft(hplus, sampling_frequency)
#     fft_hcross, _ = bilby.core.utils.nfft(hcross, sampling_frequency)
#     return {'plus': fft_hplus, 'cross': fft_hcross}
