import sys
import os
import json
import h5py
import click
import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Bidirectional, Reshape
from tensorflow.keras.utils import Sequence
from transformers import TFAutoModel, AutoConfig, AutoTokenizer
from utils import KerasCSVLogger

CACHE_DIR = 'embeddings'
model_name = 'roberta-base'
MAX_LENGTH = 512
MODELS_DIR = 'output'
SEED = 1234


class PairwiseGenerator(Sequence):

    def __init__(self, file, batch_size=32, test_mode=False):
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.hf = h5py.File(file, 'r')
        self.feats = self.hf['features']
        if test_mode is False:
            self.labels = self.hf['labels']

    def __getitem__(self, idx):
        selected = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        if self.test_mode is True:
            return self.feats[selected]
        else:
            return self.feats[selected], self.labels[selected]

    def __len__(self):
        return int(np.ceil(self.feats.shape[0]/ self.batch_size))

    def __del__(self):
        self.hf.close()


class PairwiseClassifier:

    def __init__(self, input_dim=None, dropout=0.2):
        self.dropout = dropout
        self.input_dim = input_dim
        self.model = None

    def create_model(self):
        if self.input_dim is None:
            raise Exception('Input dimension cannot be None')
        input = Input(shape=(self.input_dim,))
        x = Dense(int(self.input_dim/2), activation='relu', name="layer_1")(input)
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
        return Model(inputs=input, outputs=x)

    def train(self, train_data, n_epochs, learning_rate=0.01, valid_data=None, early_stop=5, model_name=None):
        self.model = self.create_model()
        self.model.compile(loss=tf.nn.softmax_cross_entropy_with_logits,
                           optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        self.model.summary()
        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        if model_name is None:
            model_name = "pairwise_classifier"
        log_dir = os.path.join(MODELS_DIR, model_name)
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)
        weights_path = os.path.join(log_dir, 'weights.h5')
        history = self.model.fit(train_data, epochs=n_epochs,
            validation_data=valid_data,
            callbacks=[TensorBoard(log_dir=log_dir, write_graph=False, profile_batch=0),
                       KerasCSVLogger(os.path.join(log_dir, 'training.csv'), append=True),
                       ModelCheckpoint(weights_path, save_best_only=True),
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
        self.model = load_model(os.path.join(MODELS_DIR, model_dir))

    def save_model(self, save_dir):
        save_model(self.model, save_dir)


class PairwiseBiGruClassifier(PairwiseClassifier):

    def create_model(self):
        if self.input_dim is None:
            raise Exception('Input dimension cannot be None')
        input = Input(shape=(self.input_dim,))
        x = Reshape(target_shape=(self.input_dim, 1))(input)
        x = Bidirectional(GRU(256, activation='relu', name="layer_1", return_sequences=True))(x)
        x = Dropout(self.dropout, name="dropout_1")(x)
        x = Bidirectional(GRU(64, activation='relu', name="layer_2"))(x)
        x = Dropout(self.dropout, name="dropout_2")(x)
        x = Dense(16, activation='relu', name="layer_5")(x)
        x = Dropout(self.dropout, name="dropout_5")(x)
        x = Dense(2, activation='softmax')(x)
        return Model(inputs=input, outputs=x)



@click.group()
def main():
    pass

@main.command()
@click.option('--training-data', required=True, help='Training dataset (HDF5)')
@click.option('--validation-data', required=True, help='Training dataset (HDF5)')
@click.option('--classifier-model', required=True, help='Classifier model', default='bigru',
              type=click.Choice(['simple', 'bigru'], case_sensitive=False))
@click.option('--dropout', '-d', default=0.2, help='Dropout', show_default=True, type=float)
@click.option('--learning-rate', '-r', default=0.01, help='Learning rate', show_default=True, type=float)
@click.option('--batch-size', '-s', default=32, help='Batch size', show_default=True, type=int)
@click.option('--epochs', '-e', default=20, help='Number of training epochs', show_default=True, type=int)
@click.option('--early-stop', default=5, help='Stop if no improvement is observed', show_default=True, type=int)
@click.option('--output-dir', help='Model output directory')
def train_model(training_data, validation_data, classifier_model, dropout, learning_rate, batch_size, epochs, early_stop, output_dir):
    seed(SEED)
    tf.random.set_seed(SEED)
    with h5py.File(training_data, 'r') as hf:
        input_dim = hf['features'].shape[1]
    gen_train = PairwiseGenerator(training_data, batch_size=batch_size)
    gen_valid = PairwiseGenerator(validation_data, batch_size=batch_size)
    if classifier_model == 'simple':
        classifier = PairwiseClassifier(input_dim, dropout=dropout)
    elif classifier_model == 'bigru':
        classifier = PairwiseBiGruClassifier(input_dim, dropout=dropout)
    else:
        raise Exception("Invalid classifier type")
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
@click.option('--classifier-model', required=True, help='Classifier model', default='bigru',
              type=click.Choice(['simple', 'bigru'], case_sensitive=False))
@click.option("--dictionary-file", help="JSON dictionary file with acronyms and expansions",
              default=os.path.join('data', 'diction.json'))
@click.option("--embed-full", help="Embed full sentence", is_flag=True)
def predict_sentences(test_data, model_dir, transformer_model, classifier_model, dictionary_file, embed_full):
    sys.path.extend(['features'])
    from pairwise_features import PairwiseTransformerFeatures as PairwiseFeatures
    with open(test_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:10]
    with open(dictionary_file, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(transformer_model, do_lower_case=True, add_special_tokens=False,
                                              cache_dir=CACHE_DIR, max_length=MAX_LENGTH, add_prefix_space=True)
    config = AutoConfig.from_pretrained(transformer_model)
    model = TFAutoModel.from_pretrained(transformer_model, config=config, cache_dir=CACHE_DIR)
    print(f'Embedding size: {model.config.hidden_size}')
    if classifier_model == 'simple':
        classifier = PairwiseClassifier()
    elif classifier_model == 'bigru':
        classifier = PairwiseBiGruClassifier()
    else:
        raise Exception("Invalid classifier type")
    classifier.load_model(model_dir)
    featurizer = PairwiseFeatures(tokenizer, model, dictionary, embed_full=embed_full)
    for entry in data:
        ids, sentence, expansions, feats, _ = featurizer.process_instance(entry)
        prediction, predicted_labels = classifier.predict(feats)
        result = {
            'sentence': entry['tokens'],
            'acronym': entry['tokens'][entry['acronym']],
            'prediction': expansions[int(prediction[0])],
            'score': prediction[2]
        }
        print(result)


if __name__ == "__main__":
    main()

