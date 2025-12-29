#!/usr/bin/env python
import argparse
import logging
import threading
import pyaudio
import numpy as np
from numpy.fft import fft
import os
import requests
import dotenv
import time

dotenv.load_dotenv()

LOG = logging.getLogger('krets')
LOG.addHandler(logging.StreamHandler())
LOG.handlers[-1].setFormatter(logging.Formatter('%(asctime)s - %(levelname)s :%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))


class Blinker(threading.Thread):
    def __init__(self, event, stop_event):
        super().__init__()
        self.event = event
        self.stop_event = stop_event
        self.light_id = os.getenv("LIGHT_ID")
        username = os.environ.get('HUE_USERNAME')
        address = os.environ.get('HUE_ADDRESS')
        self.base_path = f'http://{address}/api/{username}'


    def run(self):
        while not self.stop_event.is_set():
            LOG.debug("%s: Waiting for event", self.__class__.__name__)
            self.event.wait(timeout=1)
            if self.event.is_set():
                self.event.clear()
                LOG.debug("%s: event received", self.__class__.__name__)
                self.blink_light()

        LOG.debug("%s: stop_event received", self.__class__.__name__)

    def blink_light(self, blinks=4, period=0.4):
        endpoint = f'{self.base_path}/lights/{self.light_id}'
        data = requests.get(endpoint).json()
        remembered_state = data['state']

        red = {
            'on': True,
            'bri': 255,
            'hue': 255,
            'sat': 255,
            'colormode': 'hs',
            'transitiontime': 0
        }
        off = {
            'on': False,
            'transitiontime': 0
        }
        remembered_state['transitiontime'] = 0
        start_time = time.monotonic()
        for _ in range(blinks):
            for state in [red, off]:
                LOG.debug("%s: blink_light state: %s", self.__class__.__name__, state)
                requests.put(endpoint + '/state', json=state)
                while time.monotonic() - start_time < period:
                    pass
                start_time = time.monotonic()
        LOG.debug("%s: restore: %s", self.__class__.__name__, remembered_state)
        requests.put(endpoint + '/state', json=remembered_state)


class AudioAlarmDetector(threading.Thread):
    def __init__(
            self, sensitivity=0.5, tone=3300, bandwidth=100, sustain_ms=400,
            input_device_index=1, sampling_rate=44100, num_samples=2048):
        super().__init__()
        self.alarm_event = threading.Event()
        self.stop_event = threading.Event()
        self.blinker = Blinker(self.alarm_event, self.stop_event)
        self.blinker.start()

        # Configuration
        self.tone = tone
        self.bandwidth = bandwidth
        self.sensitivity = sensitivity
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples

        # Timing: How many consecutive frames must match to trigger
        frame_duration_ms = (num_samples / sampling_rate) * 1000
        self.required_consecutive = int(sustain_ms / frame_duration_ms)

        self.consecutive_hits = 0
        self.alarm_active = False
        self.input_device_index = input_device_index

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.num_samples,  # Matches read size for efficiency
            input_device_index=self.input_device_index
        )

    def detect_audio(self):
        # Blocking read eliminates the need for sleep() and reduces jitter
        try:
            raw_data = self.stream.read(self.num_samples, exception_on_overflow=False)
        except Exception as e:
            LOG.error(f"Stream read error: {e}")
            return

        audio_data = np.frombuffer(raw_data, dtype=np.int16)
        normalized_data = audio_data / 32768.0

        # Apply Hanning window to reduce spectral leakage from noise/clinks
        windowed_data = normalized_data * np.hanning(len(normalized_data))

        # FFT and Frequency Calculation
        mags = abs(fft(windowed_data))[:self.num_samples // 2]
        freqs = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate)[:self.num_samples // 2]

        peak_idx = mags.argmax()
        peak_freq = freqs[peak_idx]
        peak_mag = mags[peak_idx]

        # Calculate Noise Floor (average magnitude excluding the target band)
        mask = (freqs < self.tone - self.bandwidth) | (freqs > self.tone + self.bandwidth)
        noise_floor = np.mean(mags[mask]) if any(mask) else 0.1

        # Validation Logic
        # 1. Frequency must be within tight bandwidth
        freq_match = abs(peak_freq - self.tone) < (self.bandwidth / 2)
        # 2. Magnitude must be significantly above noise floor (SNR)
        signal_strength = peak_mag > (noise_floor + self.sensitivity)

        if freq_match and signal_strength:
            self.consecutive_hits += 1
            LOG.debug(f"Match! Freq: {peak_freq:.2f}Hz, Hits: {self.consecutive_hits}")
        else:
            self.consecutive_hits = 0

        # Trigger if sustain threshold met
        if self.consecutive_hits >= self.required_consecutive:
            if not self.alarm_active:
                LOG.info(f"Doorbell Detected: {peak_freq:.2f}Hz")
                self.alarm_event.set()
                self.alarm_active = True
                # Reset hits to prevent immediate re-triggering
                self.consecutive_hits = 0
        else:
            # Simple cool-down to reset alarm state
            if self.alarm_active and self.consecutive_hits == 0:
                self.alarm_active = False

def parse_args():
    parser = argparse.ArgumentParser('Doobell Detector')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input-index', type=int, default=1)
    parser.add_argument('--test-blink', action='store_true')
    return parser.parse_args()



def main():
    args = parse_args()
    level = logging.WARNING
    if args.debug:
        level = logging.DEBUG
    if args.verbose:
        level = logging.INFO
    LOG.setLevel(level)

    if args.test_blink:
        logging.basicConfig(level=logging.DEBUG)
        blinker = Blinker(threading.Event(), threading.Event())
        blinker.start()
        time.sleep(1)
        blinker.event.set()
        time.sleep(1)
        blinker.stop_event.set()
    else:
        detector = AudioAlarmDetector(input_device_index=args.input_index)
        detector.start()


if __name__ == "__main__":
    main()
