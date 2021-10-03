#!/bin/bash

python3 features/siamese_features.py --input-file data/train.json --output-file dataset_train_bert_siamese.h5 --embed-full \
  transformer-features --model-name bert-base-uncased
python3 features/siamese_features.py --input-file data/dev.json --output-file dataset_dev_bert_siamese.h5 --embed-full \
  transformer-features --model-name bert-base-uncased
python3 features/siamese_features.py --input-file data/test.json --output-file dataset_test_bert_siamese.h5 --embed-full \
  transformer-features --model-name bert-base-uncased  --embed-full

python3 models/siamese.py train-model --training-data dataset_train_bert_siamese.h5 --validation-data dataset_dev_bert_siamese.h5 --output-dir=bert-siamese
python3 models/siamese.py predict-sentences --test-data data/test.json --model-dir bert-siamese --transformer-model bert-base-uncased --embed-full