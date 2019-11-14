import sys
from collections import deque, namedtuple
import os
from typing import List

ArgFlag = namedtuple('ArgFlag', ('key', 'default'),
                     defaults=(None, None))

ArgParameter = namedtuple('ArgOption',
                          ('key', 'required', 'default'),
                          defaults=(None, False, ''))


def arg_parser(args, arg_desc: List[namedtuple], script_name=None):
    all_keys = {a.key for a in arg_desc}
    key_to_desc = {desc.key: desc for desc in arg_desc}
    required_keys = {a.key for a in arg_desc if isinstance(a, ArgParameter) and a.required}
    flag_keys = {a.key for a in arg_desc if isinstance(a, ArgFlag)}
    optional_keys = all_keys - required_keys - flag_keys

    script_name = script_name or os.path.basename(sys.argv[0])

    def usage(error=None):
        arg_suggest = []
        for key in flag_keys:
            arg_suggest.append(f'[--{key}]')
        for key in required_keys:
            arg_suggest.append(f'--{key} <{key}>')
        for key in optional_keys:
            arg_suggest.append(f'[--{key} <{key}>]')
        text_args_suggest = ' '.join(arg_suggest)

        if error:
            print(f'Error: {error}!')
        print('Usage:')
        print(f'    python3 {script_name} {text_args_suggest}')
        exit(0)

    if not args:
        usage()

    args = deque(args)

    results = {}
    for key in flag_keys:
        results[key] = False

    is_key = lambda k: k.startswith('--')

    current_key = None
    while args:
        arg = args.popleft()
        if current_key is None:
            if is_key(arg):
                current_key = arg[2:]
                if not current_key:
                    usage('empty key')
                if current_key not in all_keys:
                    usage(f'unknown key {arg}')
                if current_key in flag_keys:
                    results[current_key] = True
                    current_key = None
            else:
                usage('expected key name staring with --')
        else:
            if is_key(arg):
                usage('expected value without --')
            results[current_key] = arg
            current_key = None

    if not all(key in results for key in required_keys):
        missing_keys = set(required_keys) - set(results.keys())
        err_text = ', '.join(f'--{k}' for k in missing_keys)
        usage(f'not all required key filled: {err_text}')

    for key in optional_keys:
        if key not in results:
            results[key] = key_to_desc[key].default

    return results
