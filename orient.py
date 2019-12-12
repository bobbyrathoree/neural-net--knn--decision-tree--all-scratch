#!/usr/local/bin/python3

# CSCI B551 Fall 2019, Assignment 4
#
# Authors: Scott Steinbruegge (srsteinb), Bobby Rathore (brathore), Sharad Singh (singshar)

import sys
import KNN

train_or_test = sys.argv[1]
data_file_input = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]

if model.lower() == 'nearest':
    if train_or_test.lower() == 'train':
        KNN.knn_training(data_file_input, model_file)
    elif train_or_test.lower() == 'test':
        KNN.knn_testing(model_file, data_file_input)
elif model.lower() == 'tree':
    print ('Nothing here yet')
elif model.lower() == 'nnet':
    print ('Nothing here yet')
elif model.lower() == 'best':
    print ('Nothing here yet')