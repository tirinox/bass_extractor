import matplotlib.pyplot as plt
import numpy as np
from audio_helpers import amplitude, notes_dict
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


def get_note_lines_from_crepe_activation(notes, frequency, a4_freq=440):
    name_to_freq = notes_dict(a4_freq, 6)
    used_notes = {name for _, _, name in notes}
    used_notes_freq_map = {n: f for n, f in name_to_freq.items() if n in used_notes}
    return used_notes_freq_map



def plot_crepe_activation(activation, notes_xy, duration, lines_for_notes, frequencies, confidence, min_confidence):
    salience = np.flip(activation, axis=1)
    inferno = matplotlib.cm.get_cmap('inferno')
    image = inferno(salience.transpose())

    height, width, _ = image.shape

    plt.figure(figsize=(20, 10))
    plt.imshow(image)

    ticks = np.linspace(0, width, max(5, int(duration + 1.0)))
    tick_labels = [f'{x:.2f}' for x in np.linspace(0, duration, ticks.shape[0])]

    plt.xticks(ticks, tick_labels)

    for x, y, note in notes_xy:
        plt.text(x, y, note, fontsize=8, color='white')

    if frequencies is not None:
        f_x = np.arange(0, width)
        f_y = height - frequencies
        # f_y[confidence < min_confidence] = 0
        plt.plot(f_x, f_y)

    plt.yticks([])

    if lines_for_notes is not None:
        for name, v_ind in lines_for_notes.items():
            y = height - v_ind
            plt.axhline(y=y, linestyle='--', color=(1, 1, 1, 0.3))
