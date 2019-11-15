import sys
from arg_parser import ArgParameter, ArgFlag, arg_parser
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


# https://americodias.com/docs/python/audio_python.md
# https://ipython-books.github.io/116-applying-digital-filters-to-speech-sounds/

def sec_to_msec(v):
    return float(v) * 1000


def load_audio_file(input_file_path, start_time=-1, end_time=-1):
    sound = AudioSegment.from_file(file=input_file_path)
    channels = sound.channels

    if start_time < 0:
        start_time = 0

    if end_time < 0:
        end_time = sec_to_msec(sound.duration_seconds)

    sound = sound[start_time:end_time]

    print(f'Sound duration: {sound.duration_seconds:.2f}')

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
    start_time = sec_to_msec(config['start'])
    end_time = sec_to_msec(config['end'])

    samples, sample_rate = load_audio_file(input_file_path, start_time, end_time)
    mono = convert_to_mono(samples)

    if config['plot']:
        plot_audio_samples(mono, sample_rate)
        print(f'Total samples: {mono.shape[0]}')
        print(f'Min value = {np.min(mono)}, max value = {np.max(mono)}')

    f, t, Sxx = spectrogram(mono, sample_rate)

    print(f)


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('start', -1),
        ArgParameter('end', -1),
        ArgFlag('plot')
    ])
    main(config)
