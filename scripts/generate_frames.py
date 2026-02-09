import numpy as np
import bilby
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeriesDict
import utils
import gengli
import os
import argparse

"""
Script to generate *gwf frame files for ET detector.
The frames contain gaussian noise and a supernova (SN) signal.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Generate whitened gwf frame files for ET detector with gaussian noise and optional signals/glitches.")
    parser.add_argument("--seed", type=int, default=2323, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./deleteme", help="Output directory")
    parser.add_argument("--label", type=str, default="deleteme", help="Label for output files")
    parser.add_argument("--frame-duration", type=int, default=256, help="Frame duration in seconds")
    parser.add_argument("--sampling-frequency", type=int, default=4096, help="Sampling frequency in Hz")
    parser.add_argument("--minimum-frequency", type=float, default=20, help="Minimum frequency in Hz")
    parser.add_argument("--start-time", type=float, default=3600, help="Start time in seconds")
    parser.add_argument("--signal-duration", type=float, default=2, help="Signal duration in seconds")
    parser.add_argument("--inject-glitches", type=int, default=0, help="Set to 0 if you do not want to inject glitches")
    parser.add_argument("--inject-signals", type=int, help="0 will return Gaussian nosie, 1 will inject CBC, 2 will inject SN signals")
    parser.add_argument("--n-signals", type=int, default=1, help="Number of signals to inject")
    parser.add_argument("--n-glitches", type=int, default=1, help="Number of glitches to inject")
    parser.add_argument("--padding", type=float, default=5.0, help="Length of the segments near the edge where we do not inject anything")
    parser.add_argument("--plot-timeseries", type = int, default = 1, help = "Set to 1 if you want to make plots for sanity checks.")
    return parser.parse_args()


def inject_signals(args, ifos, parameters, signal_injection_times):
    if args.inject_signals == 0:
        return

    elif args.inject_signals == 1:
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

        waveform_arguments = dict(waveform_approximant="IMRPhenomXPHM", reference_frequency=50.0, minimum_frequency=args.minimum_frequency)

        waveform_generator = bilby.gw.WaveformGenerator(
            duration=args.frame_duration,
            sampling_frequency=args.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
        )
        polas = waveform_generator.frequency_domain_strain(injection_parameters)

        for ii in range(args.n_signals):
            parameters['geocent_time'] = signal_injection_times[ii]
            for ifo in ifos:
                ifo.minimum_frequency = args.minimum_frequency
                ifo.maximum_frequency = args.sampling_frequency / 2
                ifo.inject_signal_from_waveform_polarizations(injection_polarizations=polas, parameters=parameters)



    elif args.inject_signals == 2:
        polas = utils.generate_supernova_signal(target_snr=1000, duration=args.frame_duration)

        for ii in range(args.n_signals):
            parameters['geocent_time'] = signal_injection_times[ii]
            for ifo in ifos:
                ifo.minimum_frequency = args.minimum_frequency
                ifo.maximum_frequency = args.sampling_frequency / 2
                ifo.inject_signal_from_waveform_polarizations(injection_polarizations=polas, parameters=parameters)


def inject_glitches(args, ifos, generator, glitches_injection_times):
    if not args.inject_glitches:
        return
    glitchy_time_series = ifos[0].time_domain_strain
    for ii in range(args.n_glitches):
        glitchy_time_series = utils.inject_glitch(generator, glitchy_time_series, args.sampling_frequency, glitches_injection_times[ii], args.start_time, target_snr=40)

    ## Update strain data
    ifos[0].strain_data.set_from_time_domain_strain(glitchy_time_series,
                                                    sampling_frequency=args.sampling_frequency,
                                                    start_time=args.start_time, duration=args.frame_duration)


def main():
    args = parse_args()

    bilby.core.utils.random.seed(args.seed)

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    parameters = {'ra': 0.0, 'dec': 0.0, 'psi': 0.0}

    #FIX ME. The times are chosen manually. They shold be random. 
    signal_injection_times = args.start_time + np.array([11, 21])#np.random.uniform(start_time + padding, start_time+frame_duration - padding, N_signals)
    glitches_injection_times = args.start_time + np.array([15, 25])#np.random.uniform(start_time + padding, start_time+frame_duration-padding, N_glitches)
    generator = gengli.glitch_generator('L1')

    #############################################

    ifos = bilby.gw.detector.InterferometerList(["ET"])
    ifos.set_strain_data_from_zero_noise(start_time=args.start_time,
                                                       duration=args.frame_duration, sampling_frequency=args.sampling_frequency)

    inject_signals(args, ifos, parameters, signal_injection_times)

    inject_glitches(args, ifos, generator, glitches_injection_times)

    utils.save_data(filename = args.label, outdir = args.outdir, detector_network = ifos)

    if args.plot_timeseries == 1:

        data = np.load(f"./{args.outdir}/{args.label}.npz")
        t = np.arange(0, len(data['ET1']), 1)/args.sampling_frequency
        fig, axes = plt.subplots(2, 1, sharey=True, sharex=True)
        ax = axes[0]

        for a, b in zip(signal_injection_times, glitches_injection_times):
            a-=args.start_time
            b-=args.start_time
            ax.axvline(x = a, color = 'black')
            ax.axvline(x = b, color = 'black', ls = '--')

        ax.plot(t, data['ET1'])

        ax = axes[1]
        ax.plot(t, data['null_stream'])

        for xx in axes:
            xx.grid(alpha = 0.2)
        fig.savefig(f'{args.outdir}/et_{args.label}.pdf')


if __name__ == "__main__":
    main()
