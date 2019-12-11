#!/usr/local/bin/python3

# CSCI B551 Fall 2019, Assignment 4
#
# Authors: Scott Steinbruegge (srsteinb), Bobby Rathore (brathore), Sharad Singh (singshar)

import pandas as pd
import numpy as np

# Tested various K's and training set sizes. All testing results are included in our Markdown report

# Set k to best tested value
k = 30

# Import/export training data
def knn_training(training_file, model_file):
    print ("Starting KNN training")
    training_data = pd.read_csv(training_file, sep=" ", header=None)
    training_data.to_csv(model_file, index = None, header=None)
    print ("Training output generated to file", model_file)

################################################################################

def knn_testing(model_file, test_file):
    # Testing the KNN = Import the training and testing data and run KNN algorithm
    model_data = pd.read_csv(model_file, sep=",", header=None)
    test_data = pd.read_csv(test_file, sep=" ", header=None)
    
    model_data_subset = model_data.head(5000)
    
    # Create 2 data frames for testing usage
    #model_data_img = model_data.loc[:, 2:]
    model_data_img = model_data_subset.loc[:, 2:]
    test_data_img = test_data.loc[:, 2:]
   
    # Create a dataframe to store the final predictions for output
    final_preds = test_data[[0,1]].copy()
    final_preds[2] = 0
    
    counter = 0
    
    print ("Starting KNN testing")
    
    # Loop through all the test images
    for test_ind in test_data_img.index:
            # Then for every test image, loop through all training images and calculate the Euclidian distances between each of them
            
            # Make a new copy of the training data to use for every test we run
            #distance_tracking = model_data[[0, 1]].copy()
            distance_tracking = model_data_subset[[0, 1]].copy()
            distance_tracking[2] = 0
           
            # Setup the arrays to compare the test image to all of the training images
            test_data = np.array(test_data_img.iloc[test_ind,:])
            test_data = test_data.reshape(1, -1)
            train_data = np.array(model_data_img)
            
            # Calculate Euclidian distance between test image and all training images
            # Efficient L2 distance numpy arrays calculation forumula from https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c by Sourav Dey
            distance_calc = -2 * np.dot(train_data, test_data.T) + np.sum(test_data**2, axis=1) + np.sum(train_data**2, axis=1)[:, np.newaxis]
            
            # Assign distances between the test image and all training images to a data frame
            distance_tracking[2] = distance_calc
            
            # Get best guess for each orientation of the training image to the test image
            distance_tracking = distance_tracking.loc[distance_tracking.groupby(0)[2].idxmin()]
            
            # Sort the distances in ascending order and keep the top k rows by shortest distance
            distance_tracking = distance_tracking.sort_values(by=[2])
            top_k = distance_tracking.head(k)
            
            # Make a prediction based on the k images compared to and select the estimated label
            final_preds.iloc[test_ind, 2] = top_k[1].mode().iloc[0]
            
            counter += 1
            
            if counter % 100 == 0:
                print ("Testing images processed so far:", counter)
    
    pred_counts = np.where(final_preds[1] == final_preds[2], 1, 0)
    output = final_preds[[0, 2]]
    
    output.to_csv('output.txt', header=None, index=None, sep=' ', mode='a')
    
    print ("KNN predictions output saved to output.txt")
    
    print ("KNN testing accuracy:", np.round((sum(pred_counts) / len(pred_counts)) * 100, 2), "%")

# Change these to sys args when ready for submission
training_file = 'train-data.txt'
test_file = 'test-data.txt'
model_file = 'model_file.txt'

knn_training(training_file, model_file)
knn_testing(model_file, test_file)