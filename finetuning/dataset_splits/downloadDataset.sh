#!/bin/bash

# download the dataset from huggingface
mkdir dataset
cd dataset

wget https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/rev3_test_32.json
wget https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/rev3_train_32.json
wget https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/rev3_validation_32.json