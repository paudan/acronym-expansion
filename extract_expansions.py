import json
import itertools

def tuples_to_dict(tuples):
    d = dict()
    for a, b in tuples:
        d.setdefault(a, set()).add(b)
    d = {k: list(v) for k, v in d.items()}
    return d


def extract_expansions(input_files, output_file='expansions.json'):

    def extract_pairs(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [(entry['tokens'][entry['acronym']], entry['expansion']) for entry in data]

    expansions = list(itertools.chain(*map(extract_pairs, input_files)))
    expansions = tuples_to_dict(expansions)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expansions, f, indent=2)
    return expansions


def create_dataset(input_file, output_file):

    def process_entry(entry):
        expansion = entry['tokens'].copy()
        expansion[entry['acronym']] = entry['expansion']
        return {
            'acronym': entry['tokens'][entry['acronym']],
            'expansion': entry['expansion'],
            'sentence': entry['tokens'],
            'sentence_expanded': expansion
        }

    with open(input_file, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    processed = list(map(process_entry, entries))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed, f)


if __name__ == '__main__':
    import os
    # input_files = [os.path.join('data', 'train.json'), os.path.join('data', 'dev.json')]
    # extract_expansions(input_files)
    create_dataset(os.path.join('data', 'train.json'), os.path.join('features', 'train.json'))
    create_dataset(os.path.join('data', 'dev.json'), os.path.join('features', 'dev.json'))