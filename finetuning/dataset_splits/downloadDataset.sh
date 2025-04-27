#!/bin/bash

# download the dataset from huggingface
mkdir dataset && cd dataset

wget https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/test.json
wget https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/validation.json
wget https://huggingface.co/datasets/Nirmata/Movie_evaluation/resolve/main/train.json