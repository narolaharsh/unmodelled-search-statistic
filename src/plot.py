import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import scienceplots
from glob import glob
from pycbc.frame import read_frame
plt.style.use(['science'])


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DEX SNR from GWF frame files")
    parser.add_argument("--outdir", type=str, default="delete_me")
    parser.add_argument("--label", type=str, default="delete_me")
    return parser.parse_args()


def load_snr_frames(outdir, label):
    files = glob(f"{outdir}/{label}_dex_snr_*.gwf")
    dex_snr = {}
    for f in files:
        key = f.split("_dex_snr_")[-1].replace(".gwf", "")
        dex_snr[key] = read_frame(f, f"DEX:{key.upper()}")
    return dex_snr


def make_plots(args, dex_snr):

    segment_index = np.array(next(iter(dex_snr.values())).sample_times)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(10, 5))
    ax = axes[0]
    ax.scatter(segment_index, np.array(dex_snr['network_snr']), label='Network statistic (Excess power)', s=10, color='salmon')
    ax.axhline(y = 8, color = 'black', ls = '--')
    ax.scatter(segment_index, np.array(dex_snr['combined_statistic']), label='Combined statistic (Coherent excess power)', s=10, marker='x',
               color='black')


    ax = axes[1]
    if 'null_stream' in dex_snr:
        ax.scatter(segment_index, np.array(dex_snr['null_stream']), label='Null stream statistic', s=5)
    elif 'mismatch_overlap' in dex_snr:
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("Overlap")
        ax.scatter(segment_index, np.array(dex_snr['mismatch_overlap']), label='Overlap statistic', s=5)
    else:
        raise ValueError("Neither 'null_stream' nor 'mismatch_overlap' found in data")
    ax.axhline(y = 8, color = 'black', ls = '--')

    ax.set_xlabel("Seconds")

    for zz in axes:
        zz.grid(alpha = 0.2)
        zz.legend(fancybox=True, frameon=True, fontsize=8)
    fig.savefig(f"{args.outdir}/{args.label}_snr_timeseries.pdf")

    if 'null_stream' in dex_snr:
        network_snr = np.array(dex_snr['network_snr'])
        null_stream  = np.array(dex_snr['null_stream'])

        fig, ax = plt.subplots(1, 1)
        ax.scatter(network_snr, null_stream, s=5)
        ax.scatter(network_snr, network_snr - null_stream, s=5)
        ax.axvline(x = 8, color = 'black', ls = '--')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.grid()
        ax.set_aspect("equal")
        ax.set_xlabel("Network SNR")
        ax.set_ylabel("Null stream SNR")
        fig.savefig(f"{args.outdir}/{args.label}_snr_foreground_scatter.pdf")

        fig, ax = plt.subplots(1, 1)
        ax.hist(null_stream, histtype='step', cumulative=-1, density=1)
        ax.set_yscale('log')
        ax.grid()
        fig.savefig(f"{args.outdir}/{args.label}_background_histogram.pdf")


def main():
    args = parse_args()
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    dex_snr = load_snr_frames(args.outdir, args.label)
    make_plots(args, dex_snr)


if __name__ == "__main__":
    main()
