import matplotlib.pyplot as plt
import numpy as np
from audio_helpers import amplitude, notes_dict
import matplotlib.cm
from scipy.cluster.vq import vq, kmeans, whiten


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


def get_note_lines_from_crepe_activation(notes, frequency, activation, a4_freq=440):
    """
    Makes a list of lines
    :param notes: [(x, y, _)] from extract_notes
    :param frequency_list: possible frequencies on the map from crepe
    :param a4_freq: A4 note frequency (default: 440 Hz)
    :return: dict ({notename: 0..359})
    """
    name_to_freq = notes_dict(a4_freq, 6)
    used_notes = {name for _, _, name in notes}
    used_notes_freq_map = {n: f for n, f in name_to_freq.items() if n in used_notes}

    vertical_indices = np.argmax(activation, axis=1)

    # freq_to_index = {}
    # for f, index in zip(frequency, vertical_indices):
    #     if index != 0:
    #         freq_to_index[round(f, 2)] = index

    height, width, *_ = activation.shape

    f_index_pairs = np.column_stack((frequency, vertical_indices))

    codebook, distortion = kmeans(f_index_pairs, len(used_notes))

    print('codebook:')
    print(codebook)

    print('distortion:')
    print(distortion)


    f_indices = {}
    # for name, freq in used_notes_freq_map.items():
    #     best_freq = min(freq_to_index.keys(), key=lambda x: abs(x - freq))
    #     f_indices[name] = height - freq_to_index[best_freq] - 1

    return f_indices



def plot_crepe_activation(activation, notes_xy, duration, lines_for_notes):
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

    plt.yticks([])

    # for name, v_ind in lines_for_notes.items():
    #     plt.axhline(y=v_ind, linestyle='--', color='w')
