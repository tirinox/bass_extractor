from pydub import AudioSegment
from scipy.signal import spectrogram, lfilter, freqz, butter
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
