from pydub import AudioSegment
from scipy.signal import spectrogram, lfilter, freqz, butter, hilbert
import numpy as np
from math import log2, pow


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


def pitch(freq, A4=440):
    C0 = A4 * pow(2, -4.75)
    h = round(12 * log2(freq / C0))
    octave = h // 12
    n = h % 12
    return NOTE_NAMES[n] + str(octave)


def moving_average(a, n=3) :
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

    # analytic_signal = hilbert(signal)
    # amplitude_envelope = np.abs(analytic_signal)
    #
    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * frame_rate)
    # return pad_signal(amplitude_envelope, l), pad_signal(np.abs(instantaneous_frequency), l)

    return pad_signal(moving_average(np.abs(signal), window_size), signal.shape[0])


def set_to_zero_when_clipping(signal: np.ndarray, max_value):
    signal = signal.copy()
    signal[np.abs(signal) > max_value] = 0
    return signal
