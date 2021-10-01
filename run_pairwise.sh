#!/bin/bash

python3 features/pairwise_features.py --input-file data/train.json --output-file dataset_train_bert_pairwise.h5 --embed-full transformer-features --model-name bert-base-uncased
python3 features/pairwise_features.py --input-file data/dev.json --output-file dataset_dev_bert_pairwise.h5 --embed-full transformer-features --model-name bert-base-uncased
python3 features/pairwise_features.py --input-file data/test.json --output-file dataset_test_bert_pairwise.h5 --embed-full transformer-features --model-name bert-base-uncased

python3 models/pairwise.py train-model --training-data dataset_train_bert_pairwise.h5 --validation-data dataset_dev_bert_pairwise.h5 --output-dir=bert-pairwise
python3 models/pairwise.py predict-sentences --test-data data/test.json --model-dir bert-pairwise --transformer-model bert-base-uncased --embed-full