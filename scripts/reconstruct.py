import numpy as np
import sys
import torch
sys.path.append('../../deepextractor/')
from models.models import UNET2D # DeepExtractor is based on the U-Net architecture
import pickle
import sklearn
import matplotlib.pyplot as plt

def quick_snr(reconstruced_signal, data):
    return np.sqrt(np.sum(reconstruced_signal*data))


def process_segments(data_array, model, scaler, device, segment_size=8192):
    """
    Divide data into segments, reconstruct signal for each, and return SNR values.

    Args:
        data_array: Input numpy array (e.g., data['ET1'])
        model: Fine-tuned model for noise prediction
        scaler: Scaler for normalizing/denormalizing the data
        device: Device to run the model on ('cuda' or 'cpu')
        segment_size: Size of each segment (default: 8192)

    Returns:
        snr_values: List of SNR values for each segment
    """
    n_samples = len(data_array)
    n_segments = n_samples // segment_size

    snr_values = []

    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment = data_array[start_idx:end_idx]

        reconstructed = reconstruct_signal(segment, model, scaler, device)
        snr = quick_snr(reconstructed, segment)
        snr_values.append(snr)

    return np.array(snr_values)


def reconstruct_signal(input_series, model, scaler, device):
    """
    Reconstruct a glitch signal by subtracting predicted noise from the input.

    Args:
        input_series: Input time series (numpy array)
        model: Fine-tuned model for noise prediction
        scaler: Scaler for normalizing/denormalizing the data
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        g_hat: Reconstructed glitch signal (numpy array)
    """
    # STFT parameters
    n_fft = 256
    win_length = n_fft // 2
    hop_length = win_length // 2
    window = torch.hann_window(win_length)

    # Scale the input
    input_scaled = scaler.transform(input_series.reshape(-1, 1)).reshape(input_series.shape)

    # Convert to tensor and compute STFT
    signal_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    stft_result = torch.stft(
        signal_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )

    # Extract magnitude and phase
    magnitude = torch.abs(stft_result)
    phase = torch.angle(stft_result)

    # Stack and prepare for model
    stft_mag_phase = torch.stack([magnitude, phase], dim=0)
    stft_mag_phase = stft_mag_phase.unsqueeze(0)
    h_stft = stft_mag_phase.float().to(device)

    # Model prediction
    with torch.no_grad():
        n_hat_stft = model(h_stft)

    n_hat_stft = n_hat_stft.cpu()

    # Separate magnitude and phase from prediction
    magnitude_pred = n_hat_stft[:, 0, :, :]
    phase_pred = n_hat_stft[:, 1, :, :]

    # Convert back to complex STFT
    real_part = magnitude_pred * torch.cos(phase_pred)
    imag_part = magnitude_pred * torch.sin(phase_pred)
    stft_complex = torch.complex(real_part, imag_part)

    # Perform iSTFT to get time series
    n_hat_scaled = torch.istft(
        stft_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )
    n_hat_scaled = n_hat_scaled.numpy().squeeze()

    # Scale back the inverse transformed data
    n_hat = scaler.inverse_transform(
        n_hat_scaled.reshape(-1, n_hat_scaled.shape[-1])
    ).reshape(n_hat_scaled.shape)

    # Reconstruct the glitch by subtracting the predicted noise
    s_hat = input_series - n_hat

    return s_hat



model_fine_tuned = UNET2D(in_channels=2, out_channels=2)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_fine_tuned.to(device)



checkpoint_fine_tuned = torch.load("../../deepextractor/checkpoints/checkpoint_best_bilby_noise_transfer_learn.pth.tar", map_location=device, weights_only=True)
model_fine_tuned.load_state_dict(checkpoint_fine_tuned['state_dict'])
# Ensure the model is in evaluation mode
model_fine_tuned.eval()


scaler = pickle.load(open("../../deepextractor/data/scaler_bilby.pkl", 'rb'))


data = np.load(f"./deleteme/deleteme.npz")

detectors = ['ET1', 'ET2', 'ET3']


null_stream = np.zeros(len(data['ET1']))

dex_snr = {}
for i, det in enumerate(detectors):
    input_timeseries = np.asarray(data[det])
    dex_snr[det] = process_segments(input_timeseries, model_fine_tuned, scaler, device)

network_snr = np.sqrt(np.sum([dex_snr[ifo]**2 for ifo in dex_snr.keys()], axis = 0))

segment_index = np.arange(len(dex_snr['ET1']))
fig, ax = plt.subplots(1, 1)
ax.scatter(segment_index, dex_snr['ET1'])
ax.scatter(segment_index, dex_snr['ET2'])
ax.scatter(segment_index, dex_snr['ET3'])
ax.scatter(segment_index, network_snr)

fig.savefig('snr_time_series.pdf')