import sys
from arg_parser import ArgParameter, ArgFlag, arg_parser
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt


def load_audio_file(input_file_path):
    sound = AudioSegment.from_file(file=input_file_path)
    channels = sound.channels
    samples = np.array(sound.get_array_of_samples(), dtype=np.int16)
    samples = samples.reshape((int(sound.frame_count()), channels))

    return samples, sound.frame_rate


def convert_to_mono(samples):
    return np.mean(samples, axis=1)


def plot_audio_samples(samples, sampleRate, tStart=None, tEnd=None, title='Audio'):
    if not tStart:
        tStart = 0

    if not tEnd or tStart > tEnd:
        tEnd = len(samples) / sampleRate

    f, axarr = plt.subplots(2, sharex=True, figsize=(20, 10))
    axarr[0].set_title(title)
    axarr[0].plot(np.linspace(tStart, tEnd, len(samples)), samples)
    axarr[1].specgram(samples, Fs=sampleRate, NFFT=1024, noverlap=192, cmap='nipy_spectral', xextent=(tStart, tEnd))

    axarr[0].set_ylabel('Amplitude')
    axarr[1].set_ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.show()


def main(config):
    input_file_path = config['input']

    samples, sample_rate = load_audio_file(input_file_path)
    mono = convert_to_mono(samples)

    print(f'min = {np.min(mono)}, max = {np.max(mono)}')

    plot_audio_samples(mono, sample_rate)

    print(mono.shape)


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True)
    ])
    main(config)
