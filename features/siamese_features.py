import os
import json
import uuid
import h5py
import click
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors
from transformers import TFAutoModel, AutoConfig, AutoTokenizer

MODEL_DIR = 'embeddings'
model_name = 'roberta-base'
MAX_LENGTH = 512


class SiameseFeatures:

    def __init__(self, tokenizer, model, dictionary, lowercase=True, embed_full=True):
        self.tokenizer = tokenizer
        self.model = model
        self.dictionary = dictionary
        self.lowercase = lowercase
        self.embed_full = embed_full

    def embed_text(self, tokens, batched=False):
        if len(tokens) == 0:
            return tf.zeros(shape=(1, self.model.config.hidden_size))
        if batched is True:
            inputs = self.tokenizer.batch_encode_plus(tokens, return_tensors='tf', padding=True,
                                                      max_length=MAX_LENGTH, truncation=True)
        else:
            inputs = self.tokenizer.encode_plus(tokens, return_tensors='tf', padding=True, truncation=True,
                                                is_split_into_words=True, max_length=MAX_LENGTH)
        embed = self.model(**inputs)
        return embed.pooler_output

    def process_instance(self, entry):
        tokens = entry['tokens']
        if len(tokens) > 512:
            tokens = tokens[:MAX_LENGTH]
        acronym = tokens[entry['acronym']]
        expansions = self.dictionary.get(acronym)
        if expansions is None or len(expansions) == 0:
            return [], [], []
        if self.lowercase is True:
            tokens = list(map(lambda _: _.lower(), tokens))
        if self.embed_full is True:
            embed = self.embed_text(tokens)
            embed_expansions = self.embed_text(expansions, batched=True)
            inst_left = tf.repeat(embed, repeats=embed_expansions.shape[0], axis=0)
            inst_right = tf.concat([tf.repeat(embed, repeats=embed_expansions.shape[0], axis=0), embed_expansions], axis=1)
        else:
            embed_left = self.embed_text(tokens[:entry['acronym']])
            embed_right = self.embed_text(tokens[(entry['acronym']+1):])
            embed_acro = self.embed_text([tokens[entry['acronym']]])
            embed_expansions = self.embed_text(expansions, batched=True)
            inst_left = tf.repeat(tf.concat([embed_left, embed_acro, embed_right], axis=1), repeats=embed_expansions.shape[0], axis=0)
            inst_right = tf.concat([tf.repeat(embed_left, repeats=embed_expansions.shape[0], axis=0), embed_expansions,
                                    tf.repeat(embed_right, repeats=embed_expansions.shape[0], axis=0)], axis=1)
        if entry.get('expansion') is not None:
            labels = list(map(lambda exp: 1 if exp == entry['expansion'] else 0, expansions))
        else:
            labels = None
        entry_id = str(uuid.uuid4())
        ids = [entry_id] * len(expansions)
        return ids, tokens, expansions, inst_left.numpy(), inst_right.numpy(), labels

    def model_dim(self):
        transform_dim = self.model.config.hidden_size
        if self.embed_full:
            return transform_dim * 2
        else:
            return transform_dim * 3

    def create_dataset(self, data, output_file):
        f = h5py.File(output_file, mode='w')
        fdim = self.model_dim()
        idset = f.create_dataset('ids', dtype=h5py.string_dtype('ascii', 36), shape=(0,), maxshape=(None,), chunks=True)
        lset = f.create_dataset('features_left', dtype=float, shape=(0,fdim), maxshape=(None,fdim), chunks=True)
        rset = f.create_dataset('features_right', dtype=float, shape=(0,fdim), maxshape=(None,fdim), chunks=True)
        lbset = f.create_dataset('labels', dtype='i4', shape=(0,), maxshape=(None,), chunks=True)
        has_labels = False
        for entry in tqdm(data):
            ids, _, _, lfeats, rfeats, labels = self.process_instance(entry)
            idset.resize(idset.shape[0]+len(ids), axis=0)
            idset[-len(ids):] = np.array(ids, dtype=h5py.string_dtype('ascii', 36))
            lset.resize(lset.shape[0]+lfeats.shape[0], axis=0)
            lset[-lfeats.shape[0]:] = lfeats
            rset.resize(rset.shape[0]+rfeats.shape[0], axis=0)
            rset[-rfeats.shape[0]:] = rfeats
            has_labels = labels is not None
            if has_labels:
                lbset.resize(lbset.shape[0]+len(labels), axis=0)
                lbset[-len(labels):] = labels
        if has_labels is False:
            del f['labels']
        print(f'Features dataset size: {lset.shape}')
        f.close()


class SiameseWord2VecFeatures(SiameseFeatures):

    def __init__(self, dictionary, model_file, lowercase=True, embed_full=True):
        super().__init__(None, None, dictionary, lowercase=lowercase, embed_full=embed_full)
        self.model_file = model_file
        self.wv_model = KeyedVectors.load_word2vec_format(model_file, binary=False)

    @staticmethod
    def convert_glove(input_file, output_file):
        from gensim.scripts.glove2word2vec import glove2word2vec
        glove2word2vec(glove_input_file=input_file, word2vec_output_file=output_file)

    def model_dim(self):
        return self.wv_model.vector_size

    def embed_text(self, tokens, batched=False):

        def get_vec(token):
            try:
                return self.wv_model.get_vector(token)
            except KeyError:
                return [0] * self.model_dim()

        if len(tokens) == 0:
            return tf.zeros(shape=(1, self.model_dim()))
        embed = tf.expand_dims(tf.constant([get_vec(_) for _ in tokens], dtype=tf.float32), axis=0)
        return tf.math.reduce_mean(embed, axis=1)


@click.group()
@click.option("--input-file", help="JSON input file")
@click.option("--dictionary-file", help="JSON dictionary file with acronyms and expansions",
                        default=os.path.join('data', 'diction.json'))
@click.option("--embed-full", help="Embed full sentence", is_flag=True)
@click.option("--output-file", help="HDF5 output file", default='dataset.h5')
@click.pass_context
def main(ctx, input_file, dictionary_file, embed_full, output_file):
    ctx.obj["input_file"] = input_file
    ctx.obj["dictionary_file"] = dictionary_file
    ctx.obj["embed_full"] = embed_full
    ctx.obj["output_file"] = output_file

@main.command()
@click.option('--model-name', required=True, help='Transformer model name')
@click.pass_context
def transformer_features(ctx, model_name):
    with open(ctx.obj["input_file"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:1000]
    with open(ctx.obj["dictionary_file"], 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, add_special_tokens=False,
                                              cache_dir=MODEL_DIR, max_length=MAX_LENGTH, add_prefix_space=True)
    config = AutoConfig.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name, config=config, cache_dir=MODEL_DIR)
    print(f'Embedding size: {model.config.hidden_size}')
    dst = SiameseFeatures(tokenizer, model, dictionary, embed_full=ctx.obj["embed_full"])
    dst.create_dataset(data, ctx.obj["output_file"])

@main.command()
@click.option('--model-file', required=True, help='Path to GLOVE embeddings file')
@click.pass_context
def word2vec_features(ctx, model_file):
    with open(ctx.obj["input_file"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:1000]
    with open(ctx.obj["dictionary_file"], 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    dst = SiameseWord2VecFeatures(dictionary, model_file, embed_full=ctx.obj["embed_full"])
    dst.create_dataset(data, ctx.obj["output_file"])


if __name__ == '__main__':
    main(obj={})


