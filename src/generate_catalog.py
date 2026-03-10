import numpy as np
import bilby
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Generate injection parameters for CBC signals.")
    parser.add_argument("--n-events", type=int, default=100, help="Number of injection events to generate")
    parser.add_argument("--minimum-frequency", type=float, default=5, help="Minimum frequency in Hz")
    parser.add_argument("--snr-threshold", type=float, default=7, help="SNR threshold of the catalog")
    parser.add_argument("--outdir", type = str)
    parser.add_argument("--label", type = str)

    return parser.parse_args()

def setup_bbh_prior():
    prior = bilby.gw.prior.BBHPriorDict()
    prior['chirp_mass'] = bilby.core.prior.Uniform(5, 100)
    prior['mass_ratio'] = bilby.core.prior.Uniform(0.1, 1)
    prior['geocent_time'] = bilby.core.prior.Uniform(0, 24*3600)
    return prior




def compute_duration(chirp_mass, mass_ratio, minimum_frequency, safety=1.2, padding=4):
    mass_1, mass_2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(chirp_mass, mass_ratio)
    return np.ceil(padding + bilby.gw.utils.calculate_time_to_merger(minimum_frequency, mass_1, mass_2, safety=safety))


def setup_interferometers(duration, sampling_frequency, minimum_frequency, geocent_time, detector="ET"):
    waveform_arguments = dict(waveform_approximant="IMRPhenomXPHM", reference_frequency=50.0, minimum_frequency=minimum_frequency)
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    ifos = bilby.gw.detector.InterferometerList([detector])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
    start_time = geocent_time - duration + 2
    ifos.set_strain_data_from_zero_noise(start_time=start_time,
                                         duration=duration,
                                         sampling_frequency=sampling_frequency)
    return ifos, waveform_generator


def apply_snr_cut(ifos, injection_parameters, snr_threshold):
    network_matched_filter_snr = np.sum(np.array([ifo.meta_data["matched_filter_SNR"]**2 for ifo in ifos]))**0.5
    network_matched_filter_snr = np.real(network_matched_filter_snr)
    if network_matched_filter_snr > snr_threshold:
        injection_parameters['matched_filter_snr'] = network_matched_filter_snr
    else:
        target_snr = np.random.uniform(snr_threshold, 100)
        injection_parameters['luminosity_distance'] *= (network_matched_filter_snr / target_snr)
        injection_parameters['matched_filter_snr'] = target_snr

    return injection_parameters


def main():
    args = parse_args()
    prior = setup_bbh_prior()
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    output = {}

    for ii in range(args.n_events):
        _injection_parameters = prior.sample(1)
        injection_parameters = {key: float(value[0]) for key, value in _injection_parameters.items()}

        duration = compute_duration(injection_parameters['chirp_mass'],
                                    injection_parameters['mass_ratio'],
                                    args.minimum_frequency)
        
        sampling_frequency = 2048
        ifos, waveform_generator = setup_interferometers(duration, sampling_frequency,
                                                         args.minimum_frequency,
                                                         injection_parameters['geocent_time'])

        ifos.inject_signal(parameters=injection_parameters, waveform_generator=waveform_generator)

        injection_parameters = apply_snr_cut(ifos, injection_parameters, snr_threshold=args.snr_threshold)
        output[f'injection_{ii}'] = injection_parameters

    json.dump(output, open(f"{args.outdir}/{args.label}_catalog.json", 'w'), indent=4,)




        





    


if __name__ == "__main__":
    main()
