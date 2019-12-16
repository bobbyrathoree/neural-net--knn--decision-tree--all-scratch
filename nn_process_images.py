import glob
import numpy as np
from PIL import Image


def make_test_dict():
    with open(
        "/Users/bobbyrathore/Downloads/a5-photo-data/test-labels.txt", "r"
    ) as file:
        degree_mapping = {"0": 0, "90": 3, "180": 2, "270": 1}
        result = dict()
        for line in file.read().split("\n")[:-1]:
            line_element = line.split(" ")
            image = line_element[0].split("/")[-1]
            image_degree = line_element[-1]
            result.update({image: degree_mapping[image_degree]})
        return result


def load_data():
    X_train, y_train, X_test, y_test = list(), list(), list(), list()
    width, height, channels = 28, 28, 3

    # Go through train data images
    for img in glob.glob("/Users/bobbyrathore/Downloads/a5-photo-data/train/*.*"):
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
    test_mapping = make_test_dict()
    for image in glob.glob("/Users/bobbyrathore/Downloads/a5-photo-data/test/*.*"):
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

    return X_train, y_train, X_test, y_test
