import torch
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

BERT_MODEL = 'bert-base-uncased'
CACHE_DIR = '../embeddings'
INPUT_PATH = 'data'
MAX_LENGTH = 256
BATCH_SIZE = 16
MAX_EPOCHS = 5
HIDDEN_SIZE = 128
USE_CRF = False
torch.set_default_tensor_type(torch.FloatTensor)

columns = {0: 'text', 1: 'label'}
corpus = ColumnCorpus(INPUT_PATH, columns, train_file='train.txt', test_file='dev.txt', dev_file=None, tag_to_bioes=False)
corpus.filter_long_sentences(MAX_LENGTH)
embeddings = TransformerWordEmbeddings(BERT_MODEL, cache_dir=CACHE_DIR, allow_long_sentences=False)
tagger = SequenceTagger(hidden_size=HIDDEN_SIZE, embeddings=embeddings, tag_type='label',
                        tag_dictionary=corpus.make_label_dictionary(label_type='label'), use_crf=USE_CRF)
trainer = ModelTrainer(tagger, corpus)
trainer.train('acronym', learning_rate=0.1, mini_batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, embeddings_storage_mode='gpu')
