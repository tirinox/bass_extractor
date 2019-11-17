import matplotlib.pyplot as plt
import numpy as np
from audio_helpers import amplitude, find_closest_pitch
import matplotlib.cm


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


def plot_crepe_activation(activation, notes_xy, duration):
    salience = np.flip(activation, axis=1)
    inferno = matplotlib.cm.get_cmap('inferno')
    image = inferno(salience.transpose())

    plt.imshow(image)

    ticks = np.linspace(0, duration, min(100, max(5, int(duration) + 1)))
    ticks = [f'{t:.2f}' for t in ticks]

    plt.gca().set_xticklabels(['', *ticks])

    for x, y, note in notes_xy:
        plt.text(x, y, note, fontsize=8, color='white')

    plt.yticks([])

