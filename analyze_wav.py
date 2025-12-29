import wave
import numpy as np
import os


def analyze_sample(file_path):
    print(f"\n--- Analyzing: {file_path} ---")
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    with wave.open(file_path, 'rb') as wf:
        params = wf.getparams()
        sampling_rate = params.framerate
        n_frames = params.nframes

        # Read all data and convert to float
        raw_data = wf.readframes(n_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        # If stereo, take only one channel
        if params.nchannels > 1:
            audio_data = audio_data[::params.nchannels]

        # Normalize
        normalized = audio_data / 32768.0

        # Perform FFT
        n = len(normalized)
        mags = np.abs(np.fft.rfft(normalized))
        freqs = np.fft.rfftfreq(n, 1 / sampling_rate)

        # Find top 5 peaks
        indices = np.argsort(mags)[-5:][::-1]

        print(f"Sampling Rate: {sampling_rate} Hz")
        print(f"Detected Peaks (Frequency: Magnitude):")
        for idx in indices:
            print(f"  - {freqs[idx]:.2f} Hz : {mags[idx]:.4f}")


if __name__ == "__main__":
    samples = ['audio_samples/doorbell01.wav', 'audio_samples/doorbell02.wav']
    for s in samples:
        analyze_sample(s)