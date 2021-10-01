import sys
import os
import json
import h5py
import click
import numpy as np
from numpy.random import seed
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
from tensorflow.keras.utils import Sequence
from transformers import TFAutoModel, AutoConfig, AutoTokenizer
from utils import KerasCSVLogger

CACHE_DIR = 'embeddings'
model_name = 'roberta-base'
MAX_LENGTH = 512
MODELS_DIR = 'output'
SEED = 1234


class SiameseGenerator(Sequence):

    def __init__(self, file, batch_size=32, test_mode=False):
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.hf = h5py.File(file, 'r')
        self.feats_left = self.hf['features_left']
        self.feats_right = self.hf['features_right']
        if test_mode is False:
            self.labels = self.hf['labels']

    def __getitem__(self, idx):
        selected = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        if self.test_mode is True:
            return [self.feats_left[selected], self.feats_right[selected]]
        else:
            return [self.feats_left[selected], self.feats_right[selected]], self.labels[selected]

    def __len__(self):
        return int(np.ceil(self.feats_left.shape[0] / self.batch_size))

    def __del__(self):
        self.hf.close()


class SiameseClassifier:

    def __init__(self, input_dim, dropout=0.2):
        self.dropout = dropout
        self.input_dim = input_dim
        self.model = None

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    @tf.function
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def contrastive_loss_with_margin(self, margin):
        def contrastive_loss(y_true, y_pred):
            '''Contrastive loss from Hadsell-et-al.'06
            http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
            '''
            y_true = K.cast(y_true, dtype=tf.float32)
            square_pred = K.square(y_pred)
            margin_square = K.square(K.maximum(margin - y_pred, 0))
            return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
        return contrastive_loss

    def base_model(self):
        input = Input(shape=(self.input_dim,))
        x = Dense(int(self.input_dim/2), activation='relu', name="layer_1")(input)
        x = Dropout(self.dropout, name="dropout_1")(x)
        x = Dense(128, activation='relu', name="layer_2")(x)
        x = Dropout(self.dropout, name="dropout_2")(x)
        x = Dense(64, activation='relu', name="layer_3")(x)
        x = Dropout(self.dropout, name="dropout_3")(x)
        x = Dense(32, activation='relu', name="layer_4")(x)
        return Model(inputs=input, outputs=x)

    def create_model(self):
        if self.input_dim is None:
            raise Exception('Input dimension cannot be none')
        base_network = self.base_model()
        input_a = Input(shape=(self.input_dim,), name="left_input")
        vect_output_a = base_network(input_a)
        input_b = Input(shape=(self.input_dim,), name="right_input")
        vect_output_b = base_network(input_b)
        output = Lambda(self.euclidean_distance, name="distance_layer",
                        output_shape=self.eucl_dist_output_shape)([vect_output_a, vect_output_b])
        return Model([input_a, input_b], output)

    def train(self, train_data, n_epochs, learning_rate=0.01, valid_data=None, early_stop=5, model_name=None):
        self.model = self.create_model()
        self.model.compile(loss=self.contrastive_loss_with_margin(margin=1),
                           optimizer=RMSprop(learning_rate=learning_rate),
                           metrics=['accuracy'])
        self.model.summary()
        if not os.path.isdir(MODELS_DIR):
            os.mkdir(MODELS_DIR)
        if model_name is None:
            model_name = "siamese_classifier"
        log_dir = os.path.join(MODELS_DIR, model_name)
        weights_path = os.path.join(log_dir, 'weights.h5')
        history = self.model.fit(train_data, epochs=n_epochs,
            validation_data=valid_data,
            callbacks=[TensorBoard(log_dir=log_dir, write_graph=False),
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
@click.option('--dropout', '-d', default=0.2, help='Dropout', show_default=True, type=float)
@click.option('--learning-rate', '-r', default=0.01, help='Learning rate', show_default=True, type=float)
@click.option('--batch-size', '-s', default=32, help='Batch size', show_default=True, type=int)
@click.option('--epochs', '-e', default=20, help='Number of training epochs', show_default=True, type=int)
@click.option('--early-stop', default=5, help='Stop if no improvement is observed', show_default=True, type=int)
@click.option('--output-dir', help='Model output directory')
def train_model(training_data, validation_data, dropout, learning_rate, batch_size, epochs, early_stop, output_dir):
    seed(SEED)
    tf.random.set_seed(SEED)
    with h5py.File(training_data, 'r') as hf:
        input_dim = hf['features_left'].shape[1]
    gen_train = SiameseGenerator(training_data, batch_size=batch_size)
    gen_valid = SiameseGenerator(validation_data, batch_size=batch_size)
    classifier = SiameseClassifier(input_dim, dropout=dropout)
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
    sys.path.extend(['features'])
    from siamese_features import SiameseFeatures
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
    featurizer = SiameseFeatures(tokenizer, model, dictionary, embed_full=embed_full)
    classifier = SiameseClassifier(input_dim=featurizer.model_dim())
    classifier.load_model(model_dir)
    for entry in data:
        ids, sentence, expansions, lfeats, rfeats, _ = featurizer.process_instance(entry)
        prediction, predicted_labels = classifier.predict((lfeats, rfeats))
        result = {
            'sentence': entry['tokens'],
            'acronym': entry['tokens'][entry['acronym']],
            'prediction': expansions[int(prediction[0])],
            'score': prediction[2]
        }
        print(result)


if __name__ == "__main__":
    main()

