#!/usr/bin/env python

import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_samples = [
    "audio_samples/doorbell01.wav",
    "audio_samples/doorbell02.wav"
]


def analyze_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)

    # Calculate Short-Term Fourier Transform (STFT)
    hop_length = 512
    stft = np.abs(librosa.stft(y, hop_length=hop_length))

    # Extract and analyze frequencies
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2)
    mean_freqs = np.mean(stft, axis=1)
    dominant_freq_idx = np.argmax(mean_freqs)
    dominant_freq = freqs[dominant_freq_idx]

    # Duration of the audio
    duration = librosa.get_duration(y=y, sr=sr)

    # Plot a spectrogram
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        librosa.amplitude_to_db(stft, ref=np.max),
        sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.title(f"Spectrogram of {file_path}")
    plt.colorbar(format='%+2.0f dB')

    # Plot time-domain waveform
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform of {file_path}")
    plt.xlabel("Time (s)")
    plt.tight_layout()

    plt.show()

    print(f"File: {file_path}")
    print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
    print(f"Duration: {duration:.2f} seconds")

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    beeps = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units='time')
    num_beeps = len(beeps)

    print(f"Estimated Number of Beeps: {num_beeps}")


for sample in audio_samples:
    analyze_audio(sample)