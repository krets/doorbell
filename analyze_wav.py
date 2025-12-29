import librosa
import numpy as np
import scipy.signal

audio_samples = [
    "audio_samples/doorbell01.wav",
    "audio_samples/doorbell02.wav",
    "C:/Users/jesse/Downloads/test.wav"
]
def analyze_doorbell(audio_path):
    """Analyze audio file for frequency, volume, duration, and number of beeps."""

    # Load audio file
    y, sr = librosa.load(audio_path)

    # Calculate duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate volume (RMS)
    rms_volume = np.sqrt(np.mean(y ** 2))

    # Detect frequency using Short-Time Fourier Transform (STFT)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0])
    dominant_freq_idx = np.argmax(S, axis=0)
    dominant_frequency = freqs[dominant_freq_idx]
    mean_dominant_frequency = np.mean(dominant_frequency)

    # Detect number of beeps using onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    number_of_beeps = len(onset_frames)

    return {
        'duration': duration,
        'volume': rms_volume,
        'average_frequency': mean_dominant_frequency,
        'number_of_beeps': number_of_beeps
    }


def detect_doorbell(
    audio_sample,
    known_duration=1.2,
    known_volume=0.0019,
    freq_tolerance=5150,
    expected_beeps=8,
    volume_tolerance=0.001,  # Adjusted tolerance
    freq_similarity=700,     # Adjusted tolerance
    beep_tolerance=2):       # Adjusted tolerance
    """Determine if the audio sample matches the known doorbell characteristics."""

    # Load audio file
    y, sr = librosa.load(audio_sample)

    # Calculate properties
    duration = librosa.get_duration(y=y, sr=sr)
    rms_volume = np.sqrt(np.mean(y ** 2))
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=S.shape[0])
    dominant_freq_idx = np.argmax(S, axis=0)
    dominant_frequency = freqs[dominant_freq_idx]
    mean_dominant_frequency = np.mean(dominant_frequency)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    number_of_beeps = len(onset_frames)

    # Check similarities
    is_duration_similar = abs(duration - known_duration) < 0.2
    is_volume_similar = abs(rms_volume - known_volume) < volume_tolerance
    is_frequency_similar = abs(mean_dominant_frequency - freq_tolerance) < freq_similarity
    is_beep_count_similar = abs(number_of_beeps - expected_beeps) <= beep_tolerance

    return is_duration_similar and is_volume_similar and is_frequency_similar and is_beep_count_similar
# Analyze each sample
for sample in audio_samples:
    analysis = analyze_doorbell(sample)
    print(f"Analysis for {sample}:")
    print(analysis)
    print(detect_doorbell(sample))