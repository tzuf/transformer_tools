""" convert

    Every↑ rat↓ sees↑ every↑ fish↓

    to json format:

    {
        "1148_sick.matched.txt_first_arrows"
        "question" :
            {
                "stem": "A man and a woman are hiking through a wooded area"
            },
        "output": "u u u u u u u u u u u"
    }

input: a file where each sentence is in one line
output: a jsonl file with "idx", "question" and "output"

Hai Hu
"""

MY_MAP = {'↑':'u','↓':'d','=':'='}

import json, sys, os

def main():
    if len(sys.argv) != 2:
        print(' usage: python script.py fn.txt ')
        exit(0)
    convert(sys.argv[1])

def convert(fn):
    if fn[-4:] == '.txt': fn_json = fn.replace('txt','jsonl')
    else: fn_json = fn + '.jsonl'
    basename = os.path.basename(fn).replace('.txt','')
    lines = [l.strip() for l in open(fn).readlines()]
    with open(fn_json, 'w') as f:
        for i, line in enumerate(lines):
            stem = []
            output = []
            for token in line.split():
                if token[-1] not in ['↑','↓','=']:
                    print(f'some word not tagged: {line}')
                    exit()
                output.append(MY_MAP[token[-1]])
                stem.append(token[:-1])

            if len(stem) != len(output):
                print(f'len(stem) != len(output): {line}')
                exit()

            f.write(json.dumps(
                {'idx': f'{basename}-{str(i)}', 'question' : {'stem': ' '.join(stem)},
                'output': ' '.join(output) }
            ) + '\n')
    print('done!')

if __name__ == "__main__":
    main()

