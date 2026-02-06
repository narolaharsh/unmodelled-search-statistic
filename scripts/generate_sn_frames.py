import numpy as np
import bilby
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeriesDict
import utils
import gengli
import os

"""
Script to generate *gwf frame files for ET detector. 
The frames contain gaussian noise and a supernova (SN) signal. 
"""

seed = 2323
bilby.core.utils.random.seed(seed)

###### time-frequency volume, detectors #####
outdir = "./deleteme"
if not os.path.isdir(outdir):
    os.mkdir(outdir)
label = "deleteme"
frame_duration = 256
sampling_frequency = 4096
detector = 'ET'
minimum_frequency = 20
start_time = 3600
signal_duration = 2
parameters = {'ra': 0.0, 'dec': 0.0, 'psi': 0.0}
inject_sn_signals  = False
N_signals = 1
N_glitches = 2
padding = 5
signal_injection_times = start_time + np.array([11, 21])#np.random.uniform(start_time + padding, start_time+frame_duration - padding, N_signals)
glitches_injection_times = start_time + np.array([15, 25])#np.random.uniform(start_time + padding, start_time+frame_duration-padding, N_glitches)
generator = gengli.glitch_generator('L1')

inject_glitch = False
#############################################
if inject_sn_signals:
    polas = utils.generate_supernova_signal(target_snr = 1000, duration=frame_duration)

else:
    injection_parameters = dict(
    mass_1=45.0,
    mass_2=44.0,
    a_1=0.,
    a_2=0.,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=15000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    ra=parameters['ra'],
    dec=parameters['dec'])

    waveform_arguments = dict(waveform_approximant="IMRPhenomXPHM", reference_frequency=50.0, minimum_frequency=20.0)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    # the generator will convert all the parameters
    waveform_generator = bilby.gw.WaveformGenerator(
        duration = frame_duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    polas = waveform_generator.frequency_domain_strain(injection_parameters)

    

######## Frequency domain polarisations #####
#############################################

ifos = bilby.gw.detector.InterferometerList([detector])
ifos.set_strain_data_from_power_spectral_densities(start_time=start_time, duration=frame_duration, sampling_frequency=sampling_frequency)

## Inject signals in loop
for ii in range(N_signals):
    parameters['geocent_time'] = signal_injection_times[ii]
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = sampling_frequency/2
        #ifo.inject_signal_from_waveform_polarizations(injection_polarizations = polas, parameters = parameters)

## Inject glitches in loop
glitchy_time_series = ifos[0].time_domain_strain
for ii in range(N_glitches):

    glitchy_time_series = utils.inject_glitch(generator, glitchy_time_series, sampling_frequency, glitches_injection_times[ii], start_time, target_snr=40)


## Update strain data
ifos[0].strain_data.set_from_time_domain_strain(glitchy_time_series, sampling_frequency = sampling_frequency, start_time = start_time, duration = frame_duration)




utils.save_data(filename = label, outdir = outdir, detector_network = ifos)


data = np.load(f"./{outdir}/{label}.npz")
t = np.arange(0, len(data['ET1']), 1)/sampling_frequency
fig, axes = plt.subplots(2, 1, sharey=True, sharex=True)
ax = axes[0]


for val in signal_injection_times:
    ax.axvline(x = val-start_time, color = 'black')
for val in glitches_injection_times:
    ax.axvline(x = val-start_time, color = 'black', ls = '--')


ax.plot(t, data['ET1'])

ax = axes[1]
ax.plot(t, data['null_stream'])

for xx in axes:
    xx.grid(alpha = 0.2)
fig.savefig(f'et_{label}.pdf')