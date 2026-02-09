import numpy as np
import bilby
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeriesDict
import utils
import gengli
import os
import argparse
import json
import logging

"""
Script to generate *gwf frame files for ET detector.
The frames contain gaussian noise and a supernova (SN) signal.
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
    parser.add_argument("--signal-duration", type=float, default=2, help="Signal duration in seconds")
    parser.add_argument("--inject-glitches", type=int, default=0, help="Set to 0 if you do not want to inject glitches")
    parser.add_argument("--inject-signals", type=int, help="0 will return Gaussian nosie, 1 will inject CBC, 2 will inject SN signals")
    parser.add_argument("--n-signals", type=int, default=1, help="Number of signals to inject")
    parser.add_argument("--n-glitches", type=int, default=1, help="Number of glitches to inject")
    parser.add_argument("--padding", type=float, default=5.0, help="Length of the segments near the edge where we do not inject anything")
    parser.add_argument("--plot-timeseries", type = int, default = 1, help = "Set to 1 if you want to make plots for sanity checks.")
    parser.add_argument("--signal-catalog", type = str, help = 'Injection file from which the parameters should be read')
    return parser.parse_args()


def inject_signals(args, ifos, injection_catalog, signal_injection_times):
    if args.inject_signals == 0:
        return

    for ifo in ifos:
        ifo.minimum_frequency = args.minimum_frequency
        ifo.maximum_frequency = args.sampling_frequency / 2

    if args.inject_signals == 1:
        waveform_generator = bilby.gw.WaveformGenerator(
            duration=args.frame_duration,
            sampling_frequency=args.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=dict(waveform_approximant="IMRPhenomXPHM", reference_frequency=50.0, minimum_frequency=args.minimum_frequency),
        )
        for ii in range(args.n_signals):

            i_idx = np.random.randint(len(injection_catalog))
            signal_parameters = injection_catalog[f"injection_{i_idx}"]
            polas = waveform_generator.frequency_domain_strain(signal_parameters)
            signal_parameters['geocent_time'] = signal_injection_times[ii]
            for ifo in ifos:
                ifo.inject_signal_from_waveform_polarizations(injection_polarizations=polas, parameters=signal_parameters)

    elif args.inject_signals == 2:
        polas = utils.generate_supernova_signal(target_snr=1000, duration=args.frame_duration)
        for ii in range(args.n_signals):

            sky_parameters = {'geocent_time': signal_injection_times[ii], 'ra': 0.0, 'dec': 0.0, 'psi':0.0}
            for ifo in ifos:
                ifo.inject_signal_from_waveform_polarizations(injection_polarizations=polas, parameters=sky_parameters)


def inject_glitches(args, ifos, generator, glitches_injection_times):
    if not args.inject_glitches:
        return
    glitchy_time_series = ifos[0].time_domain_strain
    for ii in range(args.n_glitches):
        target_snr = np.random.uniform(7, 100, 1)
        glitchy_time_series = utils.inject_glitch(generator, glitchy_time_series, 
                                                  args.sampling_frequency, glitches_injection_times[ii], 
                                                  args.start_time, 
                                                  target_snr=target_snr)

    ## Update strain data
    ifos[0].strain_data.set_from_time_domain_strain(glitchy_time_series,
                                                    sampling_frequency=args.sampling_frequency,
                                                    start_time=args.start_time, duration=args.frame_duration)


def setup_logger(outdir, label):
    logger = logging.getLogger("generate_frames")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{outdir}/{label}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


def main():
    args = parse_args()

    bilby.core.utils.random.seed(args.seed)

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    logger = setup_logger(args.outdir, args.label)
    logger.info("Arguments: %s", vars(args))


    injection_catalog = json.load(open(args.signal_catalog, "r"))

    signal_injection_times = np.sort(np.random.uniform(args.start_time + args.padding, args.start_time+args.frame_duration - args.padding, args.n_signals))
    logger.info("Signal injection times (relative): %s", signal_injection_times - args.start_time)

    glitches_injection_times = np.sort(np.random.uniform(args.start_time + args.padding, args.start_time+args.frame_duration - args.padding, args.n_glitches))
    logger.info("Glitch injection times (relative): %s", glitches_injection_times - args.start_time)

    generator = gengli.glitch_generator('L1')

    #############################################
    

    ifos = bilby.gw.detector.InterferometerList(["ET"])
    ifos.set_strain_data_from_zero_noise(start_time=args.start_time,
                                                       duration=args.frame_duration, sampling_frequency=args.sampling_frequency)

    inject_signals(args, ifos, injection_catalog, signal_injection_times)
    logger.info("Signal injection complete (mode=%d, n_signals=%d)", args.inject_signals, args.n_signals)

    inject_glitches(args, ifos, generator, glitches_injection_times)
    logger.info("Glitch injection complete (inject_glitches=%d, n_glitches=%d)", args.inject_glitches, args.n_glitches)

    utils.save_data(filename = args.label, outdir = args.outdir, detector_network = ifos)
    logger.info("Data saved to %s/%s_frames.npz", args.outdir, args.label)

    if args.plot_timeseries == 1:

        data = np.load(f"./{args.outdir}/{args.label}_frames.npz")
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
