import matplotlib.pyplot as plt
import numpy as np
from audio_helpers import amplitude, set_to_zero_when_clipping


def plot_audio_samples(samples, sample_rate, t_start=None, t_end=None, title='Audio'):
    if not t_start:
        t_start = 0

    if not t_end or t_start > t_end:
        t_end = len(samples) / sample_rate

    print(f'Total samples: {samples.shape[0]}')
    print(f'Min value = {np.min(samples)}, max value = {np.max(samples)}')

    times = np.linspace(t_start, t_end, len(samples))
    f, axarr = plt.subplots(2, sharex=True, figsize=(20, 10))
    axarr[0].set_title(title)
    axarr[0].plot(times, samples)

    amplitude_envelope = amplitude(samples, 1000)
    axarr[0].plot(times, amplitude_envelope)

    axarr[1].specgram(samples, Fs=sample_rate, NFFT=1024, noverlap=192, cmap='nipy_spectral', xextent=(t_start, t_end))

    axarr[0].set_ylabel('Amplitude')
    axarr[1].set_ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def plot_spectrum(t, f, Sxx, lines=False):
    plt.pcolormesh(t, f, Sxx)
    if lines:
        for one_f in f[::5]:
            plt.axhline(one_f)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
