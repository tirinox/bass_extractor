import sys
from arg_parser import ArgParameter, ArgFlag, arg_parser
from pydub import AudioSegment, playback
from audio_helpers import *
from plot_audio import *



def main(config):
    input_file_path = config['input']
    start_time = sec_to_msec(config['start'])
    end_time = sec_to_msec(config['end'])

    sound = load_audio_file(input_file_path, start_time, end_time)
    sound = sound.set_channels(1)

    samples = audio_segment_to_numpy(sound)

    bass_high_fs = float(config['cutoff-high'])  # Hz
    bass_low_order = 6
    cutoff = float(config['cutoff-factor'])

    mono_lowpassed = butter_lowpass_filter(samples, cutoff, bass_high_fs, bass_low_order)

    if config['plot']:
        plot_audio_samples(mono_lowpassed, sound.frame_rate)

    time, frequency, confidence, activation = pitch_predictor(mono_lowpassed, sound.frame_rate, debug_cache=False)

    # notes = []
    # for t, f, c, _ in zip(time, frequency, confidence, activation):
    #     if c > 0.6:
    #         note = pitch(f)
    #         print(f'T = {t:.2f} sec; F = {f:.2f}, note = {note}')
    #         if not notes or notes[-1] != note:
    #             notes.append(note)
    # print(notes)
    # output = config['output']

    min_confidence = float(config['confidence'])
    a4_freq = float(config['a4'])

    plot_crepe_activation(time, frequency, confidence, activation, a4_freq, min_confidence)


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True),
        ArgParameter('start', -1),
        ArgParameter('end', -1),
        ArgParameter('cutoff-high', False, 520),
        ArgParameter('cutoff-factor', False, 1.6),
        ArgParameter('confidence', False, 0.8),
        ArgParameter('a4', False, 440),
        ArgParameter('output', False),
        ArgFlag('plot')
    ])
    main(config)
