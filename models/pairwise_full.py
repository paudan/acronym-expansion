import os
import json
import click
import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.utils import Sequence
from transformers import TFAutoModel, AutoConfig, AutoTokenizer
from utils import KerasCSVLogger, expand_dataset, expand_entry

CACHE_DIR = 'embeddings'
model_name = 'roberta-base'
MAX_LENGTH = 192
MAX_EXPAND_LENGTH = 64
MODELS_DIR = 'output'
SEED = 1234


class InputGenerator(Sequence):

    def __init__(self, data, tokenizer, batch_size=32, test_mode=False,
                 max_length=MAX_LENGTH, max_expansion_length=MAX_EXPAND_LENGTH):
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.max_expand_length = max_expansion_length

    @staticmethod
    def encode(entries, tokenizer, max_length=MAX_LENGTH, max_expansion_length=MAX_EXPAND_LENGTH):
        sentences = list(map(lambda x: x[0], entries))
        expansions = list(map(lambda x: x[1], entries))
        inputs_sent = tokenizer.batch_encode_plus(sentences, return_tensors='tf',
                is_split_into_words=True, padding='max_length', max_length=max_length, truncation=True)
        inputs_expansions = tokenizer.batch_encode_plus(expansions, return_tensors='tf',
                padding='max_length', max_length=max_expansion_length, truncation=True)
        return (inputs_sent.input_ids, inputs_sent.attention_mask,
                inputs_expansions.input_ids, inputs_expansions.attention_mask)

    def __getitem__(self, idx):
        selected = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        entries = self.data[selected]
        outputs = self.encode(entries, self.tokenizer, self.max_length)
        if self.test_mode is True:
            return outputs
        return outputs, tf.constant(list(map(lambda x: x[2], entries)), dtype=tf.int8)

    def __len__(self):
        return int(np.ceil(len(self.data)/self.batch_size))


def create_dataset(data, tokenizer, batch_size=32):

    def input_gen():
        for entry in data:
            feats = InputGenerator.encode([entry], tokenizer)
            feats = tuple([tf.reshape(feats[i], shape=(feats[i].shape[1], )) for i in range(len(feats))])
            yield feats, tf.constant([entry[2]], dtype=tf.int8)

    return tf.data.Dataset.from_generator(input_gen,
        output_types=((tf.int32, tf.int32, tf.int8, tf.int8), tf.int8),
        output_shapes=((tf.TensorShape([MAX_LENGTH]),
                        tf.TensorShape([MAX_LENGTH]),
                        tf.TensorShape([MAX_EXPAND_LENGTH]),
                        tf.TensorShape([MAX_EXPAND_LENGTH])),
                       tf.TensorShape([1]))).cache().batch(batch_size)


class PairwiseClassifier:

    def __init__(self, model_name, dropout=0.2, freeze_weights=True,
                 max_length=MAX_LENGTH, max_expansion_length=MAX_EXPAND_LENGTH):
        self.dropout = dropout
        self.model_name = model_name
        self.model = None
        self.freeze_weights = freeze_weights
        self.max_length = max_length
        self.max_expand_length = max_expansion_length

    def create_model(self):
        config = AutoConfig.from_pretrained(self.model_name)
        sentence_ids = Input(shape=(self.max_length,), dtype=tf.int32, name='input_sentence_ids')
        sentence_masks = Input(shape=(self.max_length,), dtype=tf.int8, name='input_sentence_masks')
        expand_ids = Input(shape=(self.max_expand_length,), dtype=tf.int32, name='input_expansion_ids')
        expand_masks = Input(shape=(self.max_expand_length,), dtype=tf.int8, name='input_expansion_masks')
        transformer_model = TFAutoModel.from_pretrained(self.model_name, config=config, cache_dir=CACHE_DIR)
        # Code is specific for particular model, must be edited
        # if self.freeze_weights:
        #     from transformers.models.bert.modeling_tf_bert import TFBertMainLayer
        #     for layer in transformer_model.layers[:]:
        #         if isinstance(layer, TFBertMainLayer):
        #            for idx, layer in enumerate(layer.encoder.layer):
        #                layer.trainable = False
        input_sent = transformer_model(sentence_ids, sentence_masks)
        input_expand = transformer_model(expand_ids, expand_masks)
        input_sent = input_sent[1]
        input_expand = input_expand[1]
        input = Concatenate(axis=1)([input_sent, input_expand])
        x = Dense(256, activation='relu', name="layer_1")(input)
        x = Dropout(self.dropout, name="dropout_1")(x)
        x = Dense(256, activation='relu', name="layer_2")(x)
        x = Dropout(self.dropout, name="dropout_2")(x)
        x = Dense(64, activation='relu', name="layer_3")(x)
        x = Dropout(self.dropout, name="dropout_3")(x)
        x = Dense(64, activation='relu', name="layer_4")(x)
        x = Dropout(self.dropout, name="dropout_4")(x)
        x = Dense(16, activation='relu', name="layer_5")(x)
        x = Dropout(self.dropout, name="dropout_5")(x)
        x = Dense(2, activation='softmax')(x)
        return Model(inputs=[sentence_ids, sentence_masks, expand_ids, expand_masks], outputs=x)

    def train(self, train_data, n_epochs, learning_rate=0.01, valid_data=None, early_stop=5, model_name=None):
        self.model = self.create_model()
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        self.model.summary()
        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        if model_name is None:
            model_name = "recurrent_classifier"
        log_dir = os.path.join(MODELS_DIR, model_name)
        weights_path = os.path.join(log_dir, 'weights.h5')
        history = self.model.fit(train_data, epochs=n_epochs,
            validation_data=valid_data,
            callbacks=[TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0),
                       KerasCSVLogger(os.path.join(log_dir, 'training.csv'), append=True),
                       ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True),
                       ReduceLROnPlateau(),
                       EarlyStopping(patience=early_stop)]
        )
        self.save_model(log_dir)
        return history

    def predict(self, features):
        predictions = self.model.predict(features)
        predicted_labels = np.vstack([range(predictions.shape[0]),
                                      np.argmax(predictions, axis=1),
                                      np.max(predictions, axis=1)]).T
        # Check predicted matches and return entry with highest score for predictions of 1
        matched = predicted_labels[predicted_labels[:, 1] == 1]
        if matched.shape[0] > 0:
            maxind = np.argmax(matched[:, 2])
            return tuple(matched[maxind]), predicted_labels
        # If no "matched" predictions are present, return entry with lowest score for predictions of 0
        unmatched = predicted_labels[predicted_labels[:, 1] == 0]
        if unmatched.shape[0] > 0:
            minind = np.argmin(unmatched[:, 2])
            return tuple(unmatched[minind]), predicted_labels
        return None, None

    def load_model(self, model_dir):
        self.model = self.create_model()
        weights_path = os.path.join(MODELS_DIR, model_dir, 'weights.h5')
        self.model.load_weights(weights_path, by_name=True)

    def save_model(self, save_dir):
        save_model(self.model, save_dir)


@click.group()
def main():
    pass

@main.command()
@click.option('--training-data', required=True, help='Training dataset (HDF5)')
@click.option('--validation-data', required=True, help='Training dataset (HDF5)')
@click.option("--dictionary-file", help="JSON dictionary file with acronyms and expansions",
              default=os.path.join('data', 'diction.json'))
@click.option('--transformer-model', required=True, help='Transformer model used to create features')
@click.option('--dropout', '-d', default=0.2, help='Dropout', show_default=True, type=float)
@click.option('--learning-rate', '-r', default=0.01, help='Learning rate', show_default=True, type=float)
@click.option('--batch-size', '-s', default=32, help='Batch size', show_default=True, type=int)
@click.option('--epochs', '-e', default=20, help='Number of training epochs', show_default=True, type=int)
@click.option('--early-stop', default=5, help='Stop if no improvement is observed', show_default=True, type=int)
@click.option('--output-dir', help='Model output directory')
def train_model(training_data, validation_data, dictionary_file, transformer_model,
                dropout, learning_rate, batch_size, epochs, early_stop, output_dir):
    seed(SEED)
    tf.random.set_seed(SEED)
    with open(dictionary_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    with open(training_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:1000]
    train_data = expand_dataset(data, dictionary)
    with open(validation_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:1000]
    valid_data = expand_dataset(data, dictionary)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model, do_lower_case=True, add_special_tokens=False,
                                              cache_dir=CACHE_DIR, max_length=MAX_LENGTH, add_prefix_space=True)
    # gen_train = create_dataset(train_data, tokenizer, batch_size=batch_size)
    # gen_valid = create_dataset(valid_data, tokenizer, batch_size=batch_size)
    gen_train = InputGenerator(train_data, tokenizer, batch_size=batch_size)
    gen_valid = InputGenerator(valid_data, tokenizer, batch_size=batch_size)
    classifier = PairwiseClassifier(transformer_model, dropout=dropout)
    classifier.train(gen_train, epochs,
        learning_rate=learning_rate,
        valid_data=gen_valid,
        early_stop=early_stop,
        model_name=output_dir
    )

@main.command()
@click.option('--test-data', required=True, help='Dataset for testing (JSON)')
@click.option('--model-dir', required=True, help='Model directory', default='siamese_classifier')
@click.option('--transformer-model', required=True, help='Transformer model used to create features')
@click.option("--dictionary-file", help="JSON dictionary file with acronyms and expansions",
              default=os.path.join('data', 'diction.json'))
@click.option("--embed-full", help="Embed full sentence", is_flag=True)
def predict_sentences(test_data, model_dir, transformer_model, dictionary_file, embed_full):
    with open(dictionary_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    with open(test_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:10]
    with open(dictionary_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model, do_lower_case=True, add_special_tokens=False,
                                              cache_dir=CACHE_DIR, max_length=MAX_LENGTH, add_prefix_space=True)
    classifier = PairwiseClassifier(transformer_model)
    classifier.load_model(model_dir)
    for sentence in data:
        entries = list(expand_entry(sentence, dictionary))
        expansions = list(map(lambda x: x[1], entries))
        feats = InputGenerator.encode(entries, tokenizer)
        prediction, predicted_labels = classifier.predict(feats)
        result = {
            'sentence': sentence['tokens'],
            'acronym': sentence['tokens'][sentence['acronym']],
            'prediction': expansions[int(prediction[0])],
            'score': prediction[2]
        }
        print(result)


if __name__ == "__main__":
    main()
