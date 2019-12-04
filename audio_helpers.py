from pydub import AudioSegment
from scipy.signal import lfilter, butter, spectrogram
import numpy as np
from math import log2, pow
import crepe
import os


def audio_segment_to_numpy(seg: AudioSegment):
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    return samples


def numpy_to_audio_segment(data: np.ndarray, original: AudioSegment) -> AudioSegment:
    return original._spawn(data=data.astype(dtype=np.int16))


def sec_to_msec(v):
    return float(v) * 1000


def load_audio_file(input_file_path, start_time=-1, end_time=-1) -> AudioSegment:
    sound = AudioSegment.from_file(file=input_file_path)
    channels = sound.channels

    if start_time < 0:
        start_time = 0

    if end_time < 0:
        end_time = sec_to_msec(sound.duration_seconds)

    sound = sound[start_time:end_time]

    print(f'Sound duration: {sound.duration_seconds:.2f}')
    return sound


def convert_to_mono(samples):
    return np.mean(samples, axis=1)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def c0_from_a4(a4=440):
    return a4 * pow(2, -4.75)


def note_name_from_octave_and_index(octave, n):
    return NOTE_NAMES[n] + str(octave)


def find_closest_pitch(freq, a4=440):
    c0 = c0_from_a4(a4)
    h = round(12 * log2(freq / c0))
    octave = h // 12
    n = h % 12
    return note_name_from_octave_and_index(octave, n)


def notes_dict(a4=440, octaves=5):
    c0 = c0_from_a4(a4)

    name_to_freq = {}

    for octave in range(octaves):
        c_n = 2 ** octave * c0
        c_n1 = 2 * c_n
        for note_index in range(12):
            f = (c_n1 - c_n) / 12.0 * note_index + c_n
            name_to_freq[note_name_from_octave_and_index(octave, note_index)] = round(f, 2)

    return name_to_freq


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def pad_signal(signal: np.ndarray, lenght_needed, value=0.0, front=True):
    if signal.shape[0] < lenght_needed:
        padding = np.ones((lenght_needed - signal.shape[0],), dtype=signal.dtype) * value
        return np.append(padding, signal) if front else np.append(signal, padding)
    else:
        return signal


def amplitude(signal: np.ndarray, window_size):
    return pad_signal(moving_average(np.abs(signal), window_size), signal.shape[0])


def set_to_zero_when_clipping(signal: np.ndarray, max_value):
    signal = signal.copy()
    signal[np.abs(signal) > max_value] = 0
    return signal


def extract_notes(time, freqs, confidence, activation, min_confidence, a4):
    prev_note = None
    x = 0
    height = activation.shape[0]
    for t, f, c in zip(time, freqs, confidence):
        if c > min_confidence:
            note = find_closest_pitch(f, a4)
            if note != prev_note:
                # y = height - np.argmax(activation[x], axis=0)
                y = f
                yield x, y, note
                prev_note = note

        x += 1


def pitch_predictor(signal: np.ndarray, sample_rate, model='full', step_size=10, debug_cache=False):
    def _predict():
        time, frequency, confidence, activation = crepe.predict(signal,
                                                                sample_rate,
                                                                viterbi=True,
                                                                model_capacity=model,
                                                                step_size=step_size)
        return time, frequency, confidence, activation

    predicted = False

    def _cache_name(name):
        return os.path.join(f'example/_cache_{name}.npy')

    names = 'time', 'frequency', 'confidence', 'activation'

    if debug_cache:
        if os.path.exists(_cache_name('time')):
            print('Prediction loaded from cache.')
            component_tuple = [np.load(_cache_name(name)) for name in names]
            predicted = True

    if not predicted:
        component_tuple = _predict()

    if debug_cache:
        for n, v in zip(names, component_tuple):
            np.save(_cache_name(n), v)

        print('Prediction saved cache.')

    return component_tuple


def my_pitch_predictor(mono_lowpassed, sound: AudioSegment, bass_low_fs, bass_high_fs, avg_window_size):
    freqs, times, Sxx = spectrogram(mono_lowpassed, sound.frame_rate, nfft=5000)

    freq_slice = np.where((freqs >= bass_low_fs) & (freqs <= bass_high_fs))

    freqs = freqs[freq_slice]
    Sxx = Sxx[freq_slice, :][0]

    low_sound = numpy_to_audio_segment(mono_lowpassed, sound)
    low_sound.export('example/export.wav', format='wav')

    maxes = np.argmax(Sxx, axis=0)

    amp = amplitude(mono_lowpassed, avg_window_size)
    avg_value = np.mean(amp, axis=0)
    print(f'Avg value: {avg_value:.5f}')

    for t, freq_index_of_max in zip(times, maxes):
        f = freqs[freq_index_of_max]
        frame_index = int(t * sound.frame_rate)
        current_amplitude = amp[frame_index]
        if current_amplitude > avg_value:
            print(f't = {t:.2f} sec ; note = {find_closest_pitch(f)} F = {f:.1f} hz')
        else:
            print(f't = {t:.2f} sec silent')
