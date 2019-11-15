import matplotlib.pyplot as plt
import numpy as np


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


def plot_spectrum(t, f, Sxx, lines=False):
    plt.pcolormesh(t, f, Sxx)
    if lines:
        for one_f in f[::5]:
            plt.axhline(one_f)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
