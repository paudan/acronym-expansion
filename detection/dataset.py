import os
import json
from itertools import zip_longest

def convert_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    def create_entry(entry):
        pairs = list(zip_longest(entry.get('tokens'), ['O'], fillvalue='O'))
        acro = pairs[entry.get('acronym')]
        pairs[entry.get('acronym')] = (acro[0], 'A')
        return '\n'.join(map(lambda p: p[0] + ' ' + p[1], pairs))

    with open(output_file, 'w') as f:
        for entry in data:
            f.write(create_entry(entry))
            f.writelines('\n\n')

convert_data(os.path.join('..', 'data', 'train.json'), os.path.join('data', 'train.txt'))
convert_data(os.path.join('..', 'data', 'dev.json'), os.path.join('data', 'dev.txt'))
