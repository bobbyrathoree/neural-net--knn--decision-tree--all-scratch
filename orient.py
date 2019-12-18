#!/usr/local/bin/python3

# CSCI B551 Fall 2019, Assignment 4
#
# Authors: Scott Steinbruegge (srsteinb), Bobby Rathore (brathore), Sharad Singh (singshar)

import pickle
import sys
import KNN
from adaboost import AdaBoost
from nn_layers import Dense, Activation, Dropout
from nn_model import start
from nn_process_images import load_numpy_from_pickle
from nn_utils import accuracy_score, to_categorical
from parse_text_files import parse_image_data


def save_model_to_txt(obj, file_name, model):
    with open(file_name, "w") as f:
        f.write(
            " ".join(
                [
                    "{0}.{1} = {2}".format(model, key, value)
                    for key, value in obj.__dict__.items()
                ]
            )
        )


def save_model_to_pickle(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_model_from_pickle(file_name):
    file = open(file_name, "rb")
    model_object = pickle.load(file)
    return model_object


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
    elif model in ["tree", "best"]:
        if train_or_test == "train":
            # Train model
            data_vector, all_image_ids, images_counter = parse_image_data(
                file_path=input_data_file
            )
            trained_decision_tree = AdaBoost(
                images_data_vector=data_vector,
                all_images_ids=all_image_ids,
                images_counter=images_counter,
                decision_stumps=30,
            )
            trained_decision_tree.train()
            save_model_to_pickle(obj=trained_decision_tree, file_name="tree.pkl")
            save_model_to_txt(
                obj=trained_decision_tree,
                file_name="tree_model.txt",
                model="adaboost_decision_tree",
            )

            # Test after train
            trained_decision_tree.test(test_file_path="test_file.txt")
        else:
            # Test
            trained_decision_tree = load_model_from_pickle(file_name="tree.pkl")
            trained_decision_tree.test(test_file_path="test_file.txt")

    elif model == "nnet":
        if train_or_test == "train":
            trained_neural_network = start(
                layers=[
                    Dense(300),
                    Activation("relu"),
                    Dropout(0.2),
                    Dense(300),
                    Activation("relu"),
                    Dropout(0.2),
                    Dense(4),
                    Activation("softmax"),
                ],
                epochs=35,
                learning_rate=0.001,
                rho=0.9,
                number_of_folds=5,
            )

            save_model_to_pickle(obj=trained_neural_network, file_name="nnet.pkl")
            save_model_to_txt(
                obj=trained_neural_network,
                file_name="nnet_model.txt",
                model="neural_network",
            )
        else:
            # load and test
            trained_neural_network = load_model_from_pickle(file_name="nnet.pkl")
            X_test = load_numpy_from_pickle(file_name="X_test.pkl")
            y_test = load_numpy_from_pickle(file_name="y_test.pkl")
            y_test = to_categorical(y_test, 4)

            # Get predictions
            y_pred = trained_neural_network.predict(X_test)
            score = accuracy_score(y_test, y_pred)

            print(
                "\n–– Test accuracy: {0:.2f}%\n–– Error: {1:.2f}%\n\n**************************".format(
                    100.0 * score, 100.0 * (1.0 - score)
                )
            )
    else:
        print("No such model. Choose from only nearest, tree, nnet or best.")
        exit(0)
