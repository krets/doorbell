#!/usr/bin/env python
import argparse
import logging
from datetime import datetime
import threading
import pyaudio
import numpy as np
from numpy.fft import fft
from time import sleep

LOG = logging.getLogger('krets')
LOG.addHandler(logging.StreamHandler())
LOG.handlers[-1].setFormatter(logging.Formatter('%(asctime)s - %(levelname)s :%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

class AudioAlarmDetector(threading.Thread):
    def __init__(self, sensitivity=0.5, tone=3300, bandwidth=700, beeplength=2,
                 alarmlength=1, resetlength=5, clearlength=15,
                 num_samples=2048, sampling_rate=44100, input_device_index=1):
        super().__init__()
        self.sensitivity = sensitivity
        self.tone = tone
        self.bandwidth = bandwidth
        self.beeplength = beeplength
        self.alarmlength = alarmlength
        self.resetlength = resetlength
        self.clearlength = clearlength
        self.num_samples = num_samples
        self.sampling_rate = sampling_rate
        self.input_device_index = input_device_index
        self.blipcount = 0
        self.beepcount = 0
        self.resetcount = 0
        self.clearcount = 0
        self.alarm = False
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=4096,
            input_device_index=self.input_device_index
        )

    def detect_audio(self):
        while self.stream.get_read_available() < self.num_samples:
            sleep(0.01)

        audio_data = np.frombuffer(self.stream.read(self.stream.get_read_available()), dtype=np.int16)[-self.num_samples:]
        normalized_data = audio_data / 32768.0
        intensity = abs(fft(normalized_data))[:self.num_samples // 2]
        frequencies = np.linspace(0.0, float(self.sampling_rate) / 2, num=int(self.num_samples / 2))

        which = intensity[1:].argmax() + 1
        thefreq = ((which * self.sampling_rate) / self.num_samples if which == len(intensity) - 1 else
                   ((which + (0.5 * (np.log(intensity[which + 1]) - np.log(intensity[which - 1])) /
                             (2 * np.log(intensity[which]) - np.log(intensity[which + 1]) - np.log(intensity[which - 1]))))
                    * self.sampling_rate) / self.num_samples)

        LOG.debug(f"Detected frequency: {thefreq}")

        if max(intensity[(frequencies < self.tone + self.bandwidth) & (frequencies > self.tone - self.bandwidth)]) > max(intensity[(frequencies < self.tone - 1000) & (frequencies > self.tone - 2000)]) + self.sensitivity:
            self.blipcount += 1
            self.resetcount = 0
            LOG.debug(f"Blip count: {self.blipcount}")
            if self.blipcount >= self.beeplength:
                self.blipcount = 0
                self.beepcount += 1
                LOG.debug(f"Beep count: {self.beepcount}")
                if self.beepcount >= self.alarmlength and not self.alarm:
                    self.clearcount = 0
                    self.alarm = True
                    LOG.info(f"Alarm triggered at {datetime.now()}")
                    self.beepcount = 0
        else:
            self.blipcount = 0
            self.resetcount += 1
            LOG.debug(f"Reset count: {self.resetcount}")
            if self.resetcount >= self.resetlength:
                self.resetcount = 0
                self.beepcount = 0
                if self.alarm:
                    self.clearcount += 1
                    LOG.debug(f"Clear count: {self.clearcount}")
                    if self.clearcount >= self.clearlength:
                        self.alarm = False
                        LOG.info("Alarm cleared")

    def run(self):
        LOG.info("Audio alarm detector started")
        try:
            while True:
                self.detect_audio()
                sleep(0.01)
        except KeyboardInterrupt:
            LOG.info("Audio detection stopped by user")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.pa.terminate()


def parse_args():
    parser = argparse.ArgumentParser('Doobell Detector')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input-index', type=int, default=1)
    return parser.parse_args()



def main():
    args = parse_args()
    level = logging.WARNING
    if args.debug:
        level = logging.DEBUG
    if args.verbose:
        level = logging.INFO
    LOG.setLevel(level)
    detector = AudioAlarmDetector(input_device_index=args.input_index)
    detector.start()


if __name__ == "__main__":
    main()