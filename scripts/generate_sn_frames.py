import numpy as np
import bilby
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeriesDict
import utils



"""
Script to generate *gwf frame files for ET detector. 
The frames contain gaussian noise and a supernova (SN) signal. 
"""

seed = 2323
bilby.core.utils.random.seed(seed)

###### time-frequency volume, detectors #####
outdir = "deleteme"
label = "deleteme"
duration = 2
sampling_frequency = 4096
detector = 'ET'
minimum_frequency = 20
start_time = 3600
parameters = {'ra': 0.0, 'dec': 0.0, 'geocent_time':3800, 'psi': 0.0}
supernovae_polas = utils.generate_supernova_signal(snr = 10)


print("supernovae_polas", supernovae_polas)

exit()
 ## Frequency domain polarisations
#############################################

ifos = bilby.gw.detector.InterferometerList([detector])
ifos.set_strain_data_from_power_spectral_densities(start_time=3600, duration=duration, sampling_frequency=sampling_frequency)

for ifo in ifos:
    ifo.inject_signal_from_waveform_polarizations(injection_polarizations = supernovae_polas, parameters = parameters)


utils.save_data(filename = label, outdir = outdir, detector_network = ifos)



data = TimeSeriesDict.read("deleteme/deleteme_dict.gwf", ["ET1:STRAIN", "ET2:STRAIN"])
_ = data['ET1:STRAIN']
print('data', _)