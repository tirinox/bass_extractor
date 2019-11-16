import sys
from arg_parser import ArgParameter, ArgFlag, arg_parser
from pydub import AudioSegment, playback
import numpy as np
from audio_helpers import *
from plot_audio import *


# https://americodias.com/docs/python/audio_python.md
# https://ipython-books.github.io/116-applying-digital-filters-to-speech-sounds/


def main(config):
    input_file_path = config['input']
    start_time = sec_to_msec(config['start'])
    end_time = sec_to_msec(config['end'])

    sound = load_audio_file(input_file_path, start_time, end_time)
    sound = sound.set_channels(1)

    samples = audio_segment_to_numpy(sound)

    bass_high_fs = 500.0  # sample rate, Hz
    bass_low_fs = 16.0
    bass_low_order = 6
    cutoff = 1.667

    avg_window_size = 1000

    mono_lowpassed = butter_lowpass_filter(samples, cutoff, bass_high_fs, bass_low_order)

    freqs, times, Sxx = spectrogram(mono_lowpassed, sound.frame_rate, nfft=5000)

    freq_slice = np.where((freqs >= bass_low_fs) & (freqs <= bass_high_fs))

    freqs = freqs[freq_slice]
    Sxx = Sxx[freq_slice, :][0]

    low_sound = numpy_to_audio_segment(mono_lowpassed, sound)
    low_sound.export('example/export.wav', format='wav')

    if config['plot']:
        plot_audio_samples(mono_lowpassed, sound.frame_rate)

    maxes = np.argmax(Sxx, axis=0)

    amp = amplitude(mono_lowpassed, avg_window_size)
    avg_value = np.mean(amp, axis=0)
    print(f'Avg value: {avg_value:.5f}')

    for t, freq_index_of_max in zip(times, maxes):
        f = freqs[freq_index_of_max]
        frame_index = int(t * sound.frame_rate)
        current_amplitude = amp[frame_index]
        if current_amplitude > avg_value:
            print(f't = {t:.2f} sec ; note = {pitch(f)} F = {f:.1f} hz')
        else:
            print(f't = {t:.2f} sec silent')

    # maxes[Sxx < avg] = 0



if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('start', -1),
        ArgParameter('end', -1),
        ArgFlag('plot')
    ])
    main(config)
