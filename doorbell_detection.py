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
            self.event.wait(timeout=1)
            if self.event.is_set():
                self.event.clear()
                LOG.info("Blinker: Event received, starting sequence")
                self.blink_light()

    def blink_light(self, blinks=4, period=0.4):
        endpoint = f'{self.base_path}/lights/{self.light_id}'
        try:
            data = requests.get(endpoint).json()
            remembered_state = data['state']
            red = {'on': True, 'bri': 255, 'hue': 255, 'sat': 255, 'colormode': 'hs', 'transitiontime': 0}
            off = {'on': False, 'transitiontime': 0}

            start_time = time.monotonic()
            for _ in range(blinks):
                for state in [red, off]:
                    requests.put(endpoint + '/state', json=state)
                    while time.monotonic() - start_time < period:
                        pass
                    start_time = time.monotonic()

            requests.put(endpoint + '/state', json={'on': remembered_state['on'], 'transitiontime': 0})
        except Exception as e:
            LOG.exception(f"Blinker error: {e}")


class AudioAlarmDetector(threading.Thread):
    def __init__(
            self, sensitivity=0.5, tone=3678, bandwidth=150, sustain_ms=300,
            sampling_rate=44100, num_samples=2048, input_device_name=None):

        super().__init__()
        self.alarm_event = threading.Event()
        self.stop_event = threading.Event()
        self.blinker = Blinker(self.alarm_event, self.stop_event)
        self.blinker.start()

        self.tone = tone
        self.bandwidth = bandwidth
        self.sensitivity = sensitivity
        self.sampling_rate = sampling_rate
        self.num_samples = num_samples

        frame_duration_ms = (num_samples / sampling_rate) * 1000
        self.required_consecutive = max(1, int(sustain_ms / frame_duration_ms))

        self.consecutive_hits = 0
        self.alarm_active = False
        self.pa = pyaudio.PyAudio()

        input_device_index = 1
        for i in range(self.pa.get_device_count()):
            dev_info = self.pa.get_device_info_by_index(i)
            if input_device_name and input_device_name in dev_info.get('name', ''):
                LOG.info("Selected device: %s", dev_info['name'])
                input_device_index = i
                break
        else:
            LOG.warning("Could not find device index for input name: %s", input_device_name)

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.num_samples,
            input_device_index=input_device_index
        )

    def run(self):
        LOG.info(f"Detector started. Target: {self.tone}Hz (±{self.bandwidth / 2}Hz)")
        while not self.stop_event.is_set():
            self.detect_audio()

    def detect_audio(self):
        try:
            raw_data = self.stream.read(self.num_samples, exception_on_overflow=False)
            audio_data = np.frombuffer(raw_data, dtype=np.int16)
            normalized_data = audio_data / 32768.0
            windowed_data = normalized_data * np.hanning(len(normalized_data))

            mags = abs(fft(windowed_data))[:self.num_samples // 2]
            freqs = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate)[:self.num_samples // 2]

            peak_idx = mags.argmax()
            peak_freq = freqs[peak_idx]
            peak_mag = mags[peak_idx]

            mask = (freqs < self.tone - self.bandwidth) | (freqs > self.tone + self.bandwidth)
            noise_floor = np.mean(mags[mask]) if any(mask) else 0.1
            threshold = noise_floor + self.sensitivity

            freq_match = abs(peak_freq - self.tone) < (self.bandwidth / 2)
            signal_strength = peak_mag > threshold

            # Enhanced Debugging
            if peak_mag > (threshold * 0.5):  # Only log if there is significant sound
                LOG.debug(
                    f"Freq: {peak_freq:7.1f}Hz | Mag: {peak_mag:5.2f} | Thr: {threshold:5.2f} | Match: {freq_match}")

            if freq_match and signal_strength:
                self.consecutive_hits += 1
                LOG.info(f"MATCH! Hits: {self.consecutive_hits}/{self.required_consecutive}")
            else:
                self.consecutive_hits = 0

            if self.consecutive_hits >= self.required_consecutive:
                if not self.alarm_active:
                    LOG.warning(f"🔔 DOORBELL TRIGGERED ({peak_freq:.1f}Hz)")
                    self.alarm_event.set()
                    self.alarm_active = True
                    self.consecutive_hits = 0
            else:
                if self.alarm_active and self.consecutive_hits == 0:
                    self.alarm_active = False

        except Exception as e:
            LOG.error(f"Detection loop error: {e}")


def parse_args():
    parser = argparse.ArgumentParser('Doorbell Detector')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input-name', type=str, default="hw:1,0")
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
        blinker = Blinker(threading.Event(), threading.Event())
        blinker.start()
        time.sleep(1)
        blinker.event.set()
        time.sleep(5)
        blinker.stop_event.set()
    else:
        detector = AudioAlarmDetector(input_device_name=args.input_name)
        detector.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            detector.stop_event.set()


if __name__ == "__main__":
    main()