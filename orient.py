#!/usr/local/bin/python3

# CSCI B551 Fall 2019, Assignment 4
#
# Authors: Scott Steinbruegge (srsteinb), Bobby Rathore (brathore), Sharad Singh (singshar)
import pickle
import sys
import KNN
from adaboost import AdaBoost
from parse_text_files import parse_image_data

if __name__ == "__main__":

    train_or_test = sys.argv[1].lower()
    input_data_file = sys.argv[2]
    model_file = sys.argv[3].lower()
    model = sys.argv[4]

    if model == "nearest":
        if train_or_test == "train":
            KNN.knn_training(input_data_file, model_file)
        elif train_or_test == "test":
            KNN.knn_testing(model_file, input_data_file)
    elif model == "tree":
        if train_or_test == "train":
            data_vector, all_image_ids, images_counter = parse_image_data(
                file_path=input_data_file
            )
            trainingObj = AdaBoost(
                images_data_vector=data_vector,
                all_images_ids=all_image_ids,
                images_counter=images_counter,
            )
            trainingObj.train()
            pickle.dump(trainingObj, open('training_obj', "wb"))

        print("Nothing here yet")
    elif model == "nnet":
        print("Nothing here yet")
    elif model == "best":
        print("Nothing here yet")
