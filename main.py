import sys
from arg_parser import ArgParameter, ArgFlag, arg_parser
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

    show = config['plot']
    if show:
        plot_audio_samples(mono_lowpassed, sound.frame_rate)

    time, frequency, confidence, activation = pitch_predictor(mono_lowpassed,
                                                              sound.frame_rate,
                                                              debug_cache=config['debug-cache'])

    activation = activation[:, :100]

    output = config['output']

    min_confidence = float(config['confidence'])
    a4_freq = float(config['a4'])

    add_note_lines = config['note-lines']

    notes = list(extract_notes(time, frequency, confidence, activation, min_confidence, a4_freq))

    if add_note_lines:
        note_lines = get_note_lines_from_crepe_activation(notes, frequency, a4_freq)
    else:
        note_lines = None

    plot_crepe_activation(activation, notes, sound.duration_seconds, note_lines, frequency, confidence, min_confidence)

    if output:
        plt.savefig(output, bbox_inches='tight', pad_inches=0.5)

    if show:
        plt.show()


if __name__ == '__main__':
    config = arg_parser(sys.argv[1:], [
        ArgParameter('input', True, ''),
        ArgParameter('start', False, -1),
        ArgParameter('end', False, -1),
        ArgParameter('cutoff-high', False, 520),
        ArgParameter('cutoff-factor', False, 1.6),
        ArgParameter('confidence', False, 0.65),
        ArgParameter('a4', False, 440),
        ArgParameter('output', False, ''),
        ArgFlag('debug-cache', False),
        ArgFlag('plot', True),
        ArgFlag('note-lines', True)
    ])
    main(config)
