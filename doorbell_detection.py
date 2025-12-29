#!/usr/bin/env python3
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
LOG.handlers[-1].setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s :%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))


class Blinker(threading.Thread):
    def __init__(self, event, stop_event):
        super().__init__()
        self.event, self.stop_event = event, stop_event
        self.light_id = os.getenv("LIGHT_ID")
        self.base_path = f"http://{os.getenv('HUE_ADDRESS')}/api/{os.getenv('HUE_USERNAME')}"
        self.lock = threading.Lock()

    def run(self):
        while not self.stop_event.is_set():
            if self.event.wait(timeout=1):
                self.event.clear()
                with self.lock:
                    LOG.info("Blinker: Starting sequence")
                    self.blink_light()

    def blink_light(self, blinks=4, period=0.5):
        endpoint = f'{self.base_path}/lights/{self.light_id}'
        try:
            # Capture state once
            res = requests.get(endpoint).json()
            orig_state = res['state']

            red = {'on': True, 'bri': 255, 'hue': 0, 'sat': 255, 'transitiontime': 0}
            off = {'on': False, 'transitiontime': 0}

            for _ in range(blinks):
                requests.put(f"{endpoint}/state", json=red)
                time.sleep(period)
                requests.put(f"{endpoint}/state", json=off)
                time.sleep(period)

            # Restore original state (on/off and color)
            LOG.info("Blinker: Restoring original state")
            restore = {'on': orig_state['on'], 'transitiontime': 0}
            if 'bri' in orig_state:
                restore['bri'] = orig_state['bri']
            if 'hue' in orig_state:
                restore['hue'] = orig_state['hue']
            if 'sat' in orig_state:
                restore['sat'] = orig_state['sat']

            requests.put(f"{endpoint}/state", json=restore)
        except Exception as e:
            LOG.error(f"Blinker error: {e}")


class AudioAlarmDetector(threading.Thread):
    def __init__(self, sensitivity=0.25, tones=[3153, 3678], bandwidth=250, sustain_ms=200, input_device_name="hw:1,0"):
        super().__init__()
        self.alarm_event, self.stop_event = threading.Event(), threading.Event()
        self.blinker = Blinker(self.alarm_event, self.stop_event)
        self.blinker.start()

        self.tones, self.bandwidth, self.sensitivity = tones, bandwidth, sensitivity
        self.sampling_rate, self.num_samples = 44100, 2048
        self.required_consecutive = max(1, int(sustain_ms / ((self.num_samples / self.sampling_rate) * 1000)))

        self.consecutive_hits = 0
        self.last_trigger_time = 0
        self.lockout_period = 6.0  # Seconds to wait before allowing another trigger

        self.pa = pyaudio.PyAudio()
        idx = 1
        for i in range(self.pa.get_device_count()):
            if input_device_name in self.pa.get_device_info_by_index(i).get('name', ''):
                idx = i;
                break

        self.stream = self.pa.open(
            format=pyaudio.paInt16, channels=1, rate=self.sampling_rate, input=True,
            frames_per_buffer=self.num_samples, input_device_index=idx)

    def run(self):
        LOG.info(f"Detector active. Targets: {self.tones}Hz. Min Hits: {self.required_consecutive}")
        while not self.stop_event.is_set():
            self.detect_audio()

    def detect_audio(self):
        try:
            raw = self.stream.read(self.num_samples, exception_on_overflow=False)
            mags = abs(fft(np.frombuffer(raw, dtype=np.int16) / 32768.0 * np.hanning(self.num_samples)))[
                   :self.num_samples // 2]
            freqs = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate)[:self.num_samples // 2]

            # High-pass filter check
            audible_mask = freqs > 500
            peak_idx = mags[audible_mask].argmax()
            peak_freq = freqs[audible_mask][peak_idx]
            peak_mag = mags[audible_mask][peak_idx]

            found_hit = False
            for t in self.tones:
                band_mask = (freqs > t - self.bandwidth / 2) & (freqs < t + self.bandwidth / 2)
                if any(mags[band_mask] > self.sensitivity):
                    found_hit = True
                    break

            if found_hit:
                self.consecutive_hits += 1
                LOG.debug(
                    f"MATCH! Hits: {self.consecutive_hits}/{self.required_consecutive} (Peak: {peak_freq:.1f}Hz @ {peak_mag:.2f})")
            else:
                self.consecutive_hits = 0

            # Trigger logic with lockout
            if self.consecutive_hits >= self.required_consecutive:
                now = time.time()
                if (now - self.last_trigger_time) > self.lockout_period:
                    LOG.warning("🔔 DOORBELL DETECTED")
                    self.alarm_event.set()
                    self.last_trigger_time = now
                else:
                    LOG.debug("Detection ignored (lockout active)")
                self.consecutive_hits = 0

        except Exception as e:
            LOG.error(f"Loop error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input-name', type=str, default="hw:1,0")
    args = parser.parse_args()

    LOG.setLevel(logging.DEBUG if args.debug else logging.INFO)
    detector = AudioAlarmDetector(input_device_name=args.input_name)
    detector.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detector.stop_event.set()


if __name__ == "__main__":
    main()