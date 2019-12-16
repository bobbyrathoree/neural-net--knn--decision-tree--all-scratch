import glob
from argparse import ArgumentParser

import numpy as np
from PIL import Image
import pickle

width, height, channels = 28, 28, 3


def save_numpy_to_pickle(np_array, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(np_array, f, pickle.HIGHEST_PROTOCOL)


def load_numpy_from_pickle(file_name):
    file = open(file_name, "rb")
    np_array = pickle.load(file)
    return np_array


def make_test_dict(labels_file):
    with open(labels_file, "r") as file:
        degree_mapping = {"0": 0, "90": 3, "180": 2, "270": 1}
        result = dict()
        for line in file.read().split("\n")[:-1]:
            line_element = line.split(" ")
            image = line_element[0].split("/")[-1]
            image_degree = line_element[-1]
            result.update({image: degree_mapping[image_degree]})
        return result


def load_data_from_scratch(
    train_directory: str = "/Users/bobbyrathore/Downloads/a5-photo-data/train/*.*",
    test_directory: str = "/Users/bobbyrathore/Downloads/a5-photo-data/test/*.*",
    labels_file: str = "/Users/bobbyrathore/Downloads/a5-photo-data/test-labels.txt",
):
    X_train, y_train, X_test, y_test = list(), list(), list(), list()

    # Go through train data images
    for img in glob.glob(train_directory):
        try:
            img = Image.open(img)
            img = img.resize((width, height))
            img_90 = np.array(img.rotate(90)).reshape([width, height, channels])
            img_180 = np.array(img.rotate(180)).reshape([width, height, channels])
            img_270 = np.array(img.rotate(270)).reshape([width, height, channels])
            img_0 = np.array(img).reshape([width, height, channels])
            X_train.append(img_0)
            y_train.append(0)
            X_train.append(img_90)
            y_train.append(1)
            X_train.append(img_180)
            y_train.append(2)
            X_train.append(img_270)
            y_train.append(3)
        except ValueError as ve:
            continue

    # Go through test data images
    test_mapping = make_test_dict(labels_file=labels_file)
    for image in glob.glob(test_directory):
        try:
            img = Image.open(image)
            img = img.resize((width, height))
            img = np.array(img).reshape([width, height, channels])
            X_test.append(img)
            y_test.append(test_mapping[image.split("/")[-1]])
        except ValueError as ve:
            continue

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Shuffle train data
    train_permutation = np.random.permutation(len(y_train))
    X_train = X_train[train_permutation]
    y_train = y_train[train_permutation]

    # Shuffle test data
    test_permutation = np.random.permutation(len(y_test))
    X_test = X_test[test_permutation]
    y_test = y_test[test_permutation]

    # Reshape data for neural network
    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    )
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
    )

    # Save new pickles
    save_numpy_to_pickle(X_train, "X_train.pkl")
    save_numpy_to_pickle(y_train, "y_train.pkl")
    save_numpy_to_pickle(X_test, "X_test.pkl")
    save_numpy_to_pickle(y_test, "y_test.pkl")

    return X_train, y_train, X_test, y_test


def load_data():
    X_train = load_numpy_from_pickle(file_name="X_train.pkl")
    y_train = load_numpy_from_pickle(file_name="y_train.pkl")
    X_test = load_numpy_from_pickle(file_name="X_test.pkl")
    y_test = load_numpy_from_pickle(file_name="y_test.pkl")

    return X_train, y_train, X_test, y_test


def make_train_data(train_directory):
    X_train, y_train = list(), list()

    # Go through train data images
    for img in glob.glob(train_directory):
        try:
            img = Image.open(img)
            img = img.resize((width, height))
            img_90 = np.array(img.rotate(90)).reshape([width, height, channels])
            img_180 = np.array(img.rotate(180)).reshape([width, height, channels])
            img_270 = np.array(img.rotate(270)).reshape([width, height, channels])
            img_0 = np.array(img).reshape([width, height, channels])
            X_train.append(img_0)
            y_train.append(0)
            X_train.append(img_90)
            y_train.append(1)
            X_train.append(img_180)
            y_train.append(2)
            X_train.append(img_270)
            y_train.append(3)
        except ValueError as ve:
            continue

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Shuffle train data
    train_permutation = np.random.permutation(len(y_train))
    X_train = X_train[train_permutation]
    y_train = y_train[train_permutation]

    # Reshape data for neural network
    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    )

    # Save new pickles
    save_numpy_to_pickle(X_train, "X_train.pkl")
    save_numpy_to_pickle(y_train, "y_train.pkl")


def make_test_data(test_directory, labels_file):
    X_test, y_test = list(), list()

    # Go through test data images
    test_mapping = make_test_dict(labels_file=labels_file)
    for image in glob.glob(test_directory):
        try:
            img = Image.open(image)
            img = img.resize((width, height))
            img = np.array(img).reshape([width, height, channels])
            X_test.append(img)
            y_test.append(test_mapping[image.split("/")[-1]])
        except ValueError as ve:
            continue

    # Convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Shuffle test data
    test_permutation = np.random.permutation(len(y_test))
    X_test = X_test[test_permutation]
    y_test = y_test[test_permutation]

    # Reshape data for neural network
    X_test = X_test.reshape(
        X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]
    )

    # Save new pickles
    save_numpy_to_pickle(X_test, "X_test.pkl")
    save_numpy_to_pickle(y_test, "y_test.pkl")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", "-t", type=str, default=None, required=False)
    parser.add_argument("--test", "-s", type=str, default=None, required=False)
    parser.add_argument("--labels", "-l", type=str, default=None, required=False)
    args = parser.parse_args()

    if args.train and not args.test:
        make_train_data(train_directory="{0}/*.*".format(args.train))
    elif args.test and not args.train:
        make_test_data(
            test_directory="{0}/*.*".format(args.test), labels_file=args.labels
        )
    elif args.train and args.test:
        load_data_from_scratch(
            train_directory="{0}/*.*".format(args.train),
            test_directory="{0}/*.*".format(args.test),
            labels_file=args.labels,
        )
    else:
        print("No parameters entered.")
        exit(0)
