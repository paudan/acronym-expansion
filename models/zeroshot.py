import os
import json
import argparse
from tqdm import tqdm
from transformers import pipeline, AutoModel, AutoTokenizer

model_name = 'facebook/bart-large-mnli'
MODEL_DIR = 'embeddings'
MAX_LENGTH = 512
DEVICE = -1  # CPU

def process_entry(entry, dictionary):
    tokens = entry['tokens']
    acronym = tokens[entry['acronym']]
    expansions = dictionary.get(acronym)
    if expansions is None or len(expansions) == 0:
        return None, None
    tokens = ' '.join(map(lambda _: _.lower(), tokens))
    return tokens, expansions

def check_prediction(classifier, entry, dictionary):
    tokens, expansions = process_entry(entry, dictionary)
    actual = entry.get('expansion')
    pred = classifier(tokens, expansions)
    predicted = pred.get('labels')
    if not predicted or not isinstance(predicted, list):
        return False
    return actual == predicted[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", help="JSON input file")
    parser.add_argument("--dictionary-file", help="JSON dictionary file with acronyms and expansions",
                        default=os.path.join('data', 'diction.json'))
    parser.add_argument("--model-name", help="Model name", default=model_name)
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:1000]
    with open(args.dictionary_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    model_name = args.model_name
    model = AutoModel.from_pretrained(model_name, cache_dir=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_DIR)
    classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer, device=DEVICE)
    results = list(map(lambda x: check_prediction(classifier, x, dictionary), tqdm(data)))
    print('Accuracy:', sum(results)/len(results))
