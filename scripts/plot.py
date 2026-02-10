import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import scienceplots
plt.style.use(['science'])


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DEX SNR from saved npz file")
    parser.add_argument("--snr-data", type=str, required=True, help="Path to dex_snr.npz file")
    parser.add_argument("--outdir", type=str, default="deleteme")
    parser.add_argument("--label", type=str, default="deleteme")
    return parser.parse_args()


def make_plots(args, dex_snr):
    first_key = list(dex_snr.keys())[0]
    segment_index = np.arange(len(dex_snr[first_key])) * 2

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5))
    ax = axes[0]
    ax.scatter(segment_index, dex_snr['network_snr'], label='Network statistic (Excess power)', s=10, color='salmon')
    ax.axhline(y = 8, color = 'black', ls = '--')
    ax.scatter(segment_index, dex_snr['combined_statistic'], label='Combined statistic (Coherent excess power)', s=10, marker='x',
               color='black')

    
    ax = axes[1]
    if 'null_stream' in dex_snr:
        ax.scatter(segment_index, dex_snr['null_stream'], label='Null stream statistic', s=5)
    elif 'mismatch_overlap' in dex_snr:
        ax.scatter(segment_index, dex_snr['mismatch_overlap'], label='Mismatch overlap statistic', s=5)
    else:
        raise ValueError("Neither 'null_stream' nor 'mismatch_overlap' found in data")
    ax.axhline(y = 8, color = 'black', ls = '--')

    ax.set_xlabel("Seconds")

    for zz in axes:
        zz.grid(alpha = 0.2)
        zz.set_ylabel("DE SNR")
        zz.legend(fancybox=True, frameon=True, fontsize=8)
    fig.savefig(f"{args.outdir}/{args.label}_snr_timeseries.pdf")

    if 'null_stream' in dex_snr:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(dex_snr['network_snr'], dex_snr['null_stream'], s = 5)
        ax.scatter(dex_snr['network_snr'], dex_snr['network_snr']-dex_snr['null_stream'], s = 5)
        #ax.scatter(dex_snr['network_snr'], dex_snr['combined_statistic'], label='signals')
        ax.axvline(x = 8, color = 'black', ls = '--')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.grid()
        ax.set_aspect("equal")
        ax.set_xlabel("Network SNR")
        ax.set_ylabel("Null stream SNR")
        fig.savefig(f"{args.outdir}/{args.label}_snr_foreground_scatter.pdf")



def main():
    args = parse_args()
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    dex_snr = dict(np.load(args.snr_data))
    make_plots(args, dex_snr)


if __name__ == "__main__":
    main()
