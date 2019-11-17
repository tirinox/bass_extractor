import matplotlib.pyplot as plt
import numpy as np
from audio_helpers import amplitude, pitch
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


def plot_crepe_activation(time, freqs, confidence, activation, a4=440, min_confidence=0.8):
    salience = np.flip(activation, axis=1)
    inferno = matplotlib.cm.get_cmap('inferno')
    image = inferno(salience.transpose())

    plt.imshow(image)

    prev_note = None
    x = 0
    height = image.shape[0]
    for t, f, c in zip(time, freqs, confidence):
        if c > min_confidence:
            note = pitch(f, a4)
            if note != prev_note:
                y = height - np.argmax(activation[x]) - 20 - 20 * (x % 4)
                plt.text(x, y, note, fontsize=8, color='white')
                prev_note = note
        else:
            prev_note = None
        x += 1

    plt.show()