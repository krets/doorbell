#!/usr/bin/env python
from datetime import datetime
import pyaudio
import numpy as np
from numpy.fft import fft
from time import sleep

SENSITIVITY = 0.5  # Try different values between 0.1 and 1
TONE = 3300
BANDWIDTH = 700  # Increase this if needed
beeplength = 2  # Reduce this if your doorbell rings are shorter
alarmlength = 1  # Reduce this if you want faster alarm triggering
resetlength = 5  # Adjust as needed
clearlength = 15  # Adjust as needed
debug = False
frequencyoutput = True

NUM_SAMPLES = 2048
SAMPLING_RATE = 44100
pa = pyaudio.PyAudio()

# List Devices
for i in range(pa.get_device_count()):
    print(pa.get_device_info_by_index(i))

_stream = pa.open(format=pyaudio.paInt16,
                  channels=1,
                  rate=SAMPLING_RATE,
                  input=True,
                  frames_per_buffer=4096,
                  input_device_index=1)

print("Alarm detector working. Press CTRL-C to quit.")

blipcount = 0
beepcount = 0
resetcount = 0
clearcount = 0
alarm = False

try:
    while True:
        while _stream.get_read_available() < NUM_SAMPLES:
            sleep(0.01)
        audio_data = np.frombuffer(_stream.read(
            _stream.get_read_available()), dtype=np.int16)[-NUM_SAMPLES:]
        normalized_data = audio_data / 32768.0
        intensity = abs(fft(normalized_data))[:NUM_SAMPLES//2]
        frequencies = np.linspace(0.0, float(SAMPLING_RATE)/2, num=int(NUM_SAMPLES/2))

        if frequencyoutput:
            which = intensity[1:].argmax() + 1
            if which != len(intensity) - 1:
                y0, y1, y2 = np.log(intensity[which-1:which+2:])
                x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
                thefreq = (which + x1) * SAMPLING_RATE / NUM_SAMPLES
            else:
                thefreq = which * SAMPLING_RATE / NUM_SAMPLES
            if debug: print("\t\t\t\tfreq=", thefreq)

        if max(intensity[(frequencies < TONE+BANDWIDTH) & (frequencies > TONE-BANDWIDTH)]) > max(intensity[(frequencies < TONE-1000) & (frequencies > TONE-2000)]) + SENSITIVITY:
            blipcount += 1
            resetcount = 0
            if debug: print("\t\tBlip", blipcount)
            if blipcount >= beeplength:
                blipcount = 0
                resetcount = 0
                beepcount += 1
                if debug: print("\tBeep", beepcount)
                if beepcount >= alarmlength and not alarm:
                    clearcount = 0
                    alarm = True
                    print("Alarm!", datetime.now())
                    beepcount = 0
        else:
            blipcount = 0
            resetcount += 1
            if debug: print("\t\t\treset", resetcount)
            if resetcount >= resetlength:
                resetcount = 0
                beepcount = 0
                if alarm:
                    clearcount += 1
                    if debug: print("\t\tclear", clearcount)
                    if clearcount >= clearlength:
                        clearcount = 0
                        print("Cleared alarm!")
                        alarm = False
        sleep(0.01)
except KeyboardInterrupt:
    print("User interrupted the operation.")
finally:
    _stream.stop_stream()
    _stream.close()
    pa.terminate()