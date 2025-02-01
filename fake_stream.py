import pyaudio
import wave
import os

WAVE_FILES_DIRECTORY = 'audio_samples'

2
def list_wave_files(directory):
    return [f for f in os.listdir(directory) if f.endswith('.wav')]


def play_wave_file(filename, output_device_index=None):
    chunk = 1024
    wf = wave.open(filename, 'rb')

    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pa.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True,
        output_device_index=output_device_index)

    data = wf.readframes(chunk)

    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    wf.close()
    pa.terminate()


def main_menu(files):
    print("Select a wave file to play:")
    for idx, file in enumerate(files):
        print(f"{idx}: {file}")
    choice = int(input("Enter your choice: "))
    return files[choice%len(files)]


def main():
    wave_files = list_wave_files(WAVE_FILES_DIRECTORY)
    if not wave_files:
        print("No wave files found in the specified directory.")
        return

    output_device_index = 10  # Use None for default

    while True:
        selected_file = main_menu(wave_files)
        print(f"Playing {selected_file}...")
        play_wave_file(os.path.join(WAVE_FILES_DIRECTORY, selected_file), output_device_index)


if __name__ == '__main__':
    main()