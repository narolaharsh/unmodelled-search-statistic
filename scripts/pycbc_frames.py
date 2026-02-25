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



def generate_noise(args):
    """
    Generate coloured noise using PyCBC
    """
    
    delta_f = 1.0/8
    length = int(args.sampling_frequency/2/delta_f) + 1
    _scripts_dir = os.path.dirname(os.path.abspath(__file__))

    if args.detector_network == "ETT":
        psd_file = './noise_curves/ET_D_psd.txt'
        detectors = ["ET1", "ET2", "ET3"]
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
        write_frame(f"{args.outdir}/{args.label}_{det}_duration_{args.frame_duration}.gwf", f"{det}:STRAIN", ts)
        noise_timeseries[det] = ts

    return noise_timeseries


def main():

    args = parse_args()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)


    noise_timeseries = generate_noise(args)

    if args.plot_timeseries:
        n_det = len(noise_timeseries)
        fig, axes = plt.subplots(n_det, 1, figsize=(10, 3 * n_det), sharex=True)
        if n_det == 1:
            axes = [axes]
        for ax, (det, ts) in zip(axes, noise_timeseries.items()):
            ax.plot(ts.sample_times, ts)
            ax.set_ylabel(f"{det} Strain")
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/{args.label}.pdf")







if __name__ == "__main__":
    main()