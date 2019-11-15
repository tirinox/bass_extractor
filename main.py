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

    if config['plot']:
        plot_audio_samples(samples, sound.frame_rate)
        print(f'Total samples: {samples.shape[0]}')
        print(f'Min value = {np.min(samples)}, max value = {np.max(samples)}')

    bass_high_fs = 500.0  # sample rate, Hz
    bass_low_fs = 16.0
    bass_low_order = 6
    cutoff = 3.667

    mono_lowpassed = butter_lowpass_filter(samples, cutoff, bass_high_fs, bass_low_order)

    f, t, Sxx = spectrogram(mono_lowpassed, sound.frame_rate, nfft=5000)

    freq_slice = np.where((f >= bass_low_fs) & (f <= bass_high_fs))

    f = f[freq_slice]
    Sxx = Sxx[freq_slice, :][0]

    print(f)
    print(Sxx.shape)

    low_sound = numpy_to_audio_segment(mono_lowpassed, sound)
    low_sound.export('example/export.wav', format='wav')

    # plot_spectrum(t, f, Sxx)

    maxes = np.argmax(Sxx, axis=0)

    plot_audio_samples(mono_lowpassed, sound.frame_rate)

    for m in maxes:
        freq = f[m]
        print(pitch(m), f'{freq} hz')

    # avg = np.mean(np.abs(mono_lowpassed), axis=0)
    # print(avg)

    # maxes[Sxx < avg] = 0

    # plt.plot(t, maxes)
    # plt.show()

    # print(str(maxes))



if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('start', -1),
        ArgParameter('end', -1),
        ArgFlag('plot')
    ])
    main(config)
