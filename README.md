# a4

## KNN

For our KKN implementation, we have two functions, one to handle the training data and one to generate predictions on the test data. The training function accepts the input training data file and output model file parameters specified when the program is executed. The training for KNN just consists of saving the entire set of training data to the model file for use later on in the testing function. The majority of the work is performed within the testing function. 

The test function accepts model input file and test data input file as parameters. The model file is the one that was generated by the training function. Both the test data and model data are imported into pandas data frames. Pandas was used so it'd be easier to perform data manipulation in the subsequent steps. Next, a subset of the training data is chosen to use instead of the full training set of data. This was done because it allowed the testing to run much faster than using the entire training data set and only resulted in a 0.8% decrease in prediction accuracy for a 14 minute decrease in runtime. The subset that worked the best during our testing was a 5000 row training data set. We tried numerous combinations of k and training data sizes and this combination of k = 30 and n = 5000 provided the best balance of both performance and accuracy. Below is a table of the combinations of k and training set size that were tested with corresponding run times. 

| k                  | n=500  | n=2500 | n=5000 | n=7500 | n=15000 | n=Full |
|--------------------|--------|--------|--------|--------|---------|--------|
| 5                  | 63.63% |        | 67.23% |        |         | 69.25% |
| 10                 | 65.75% |        | 68.82% |        |         | 71.26% |
| 25                 | 68.29% |        | 70.52% |        |         | 71.79% |
| 30                 | 68.93% | 69.78% | 70.63% | 70.41% | 70.63%  | 72.11% |
| 50                 |        |        | 70.31% |        |         |        |
| 100                | 67.97% |        | 70.1%  |        |         | 71.16% |
| Runtime for k (mi) | < 1    | 1      | 2      | 3      | 6       | 15     |

The size of k had no impact on the overall runtime for a training set size since the distance calculations were already performed before it took the k best into consideration later on. The only thing that really impacted overall test runtimes was the size of the training set. The full set obviously provided the highest accuracy overall, but not by much, so we decided to keep the runtime low for a slight hit to accuracy.

After the subset of the training data is assigned to a data frame, two more variables are created to hold just the image pixel values for the training and test data each. Once all the data we needed was imported and in place, we could then start looping through each of the images in the test set. Inside this loop, we first create a distance tracking dataframe to hold the values generated when calulcating the Euclidian distance between the current test image and all training images. Then the data frames that hold the image data needed to be reshaped and converted to numpy arrays in order to perform vectorized distance calculations between them. A distance forumla is then applied to these two vectors to calculate Euclidian distance between the test image and every image in the training data set. These values are then assigned to the distance tracking dataframe created earlier, where we could then forumate a best guess for each training images orientation compared to the test image based on this distance value. Once we had our best guesses, we could then sort the distance tracking data frame in ascending order and select the top k records of the data frame based on the k value we specified at runtime. The mode of the top k predictions are then selected which will then represent our image orientation prediction which then gets saved to a final predictions data frame.

Once all of the test images have had a prediction made for them, we output the predictions to an output.txt file and then check to see how many predictions we got right compared to the true class label. By far the biggest challenge for this assignment was to get distance calculations between each training image and all of the images in the training data set to run quickly. We tried to vectorize the calculation, but couldn't figure out how to generate predictions for all test images versus all training images instead of looping through each test image one at a time. The best solution that provided a balance of runtime and accuracy was use less of the training data and find that sweet spot that ran quickly and still provided good predictions.


## Neural Network
#### Please note that after [consulting with David](https://drive.google.com/file/d/19_FEnIYULNgX4wjnQWTzkmY-pXHR1TB-/view?usp=sharing) and forwarding the case to entire 551 staff, the code for this model and it's markdown report will be the same for both Kelly-Neha's repository and this repository.
To build the neural network from scratch, we used the model Bobby had built for his 556 assignment that required him to build a model similar [to this](https://keras.io/examples/mnist_mlp/) that included Dense, Activation and Dropout layers with RMSProp optimization. The primary challenge was to figure out a way to preprocess the training images, since we had only 10000 images to work with. Even though those 10000 images could be rotated in 3 other normal degrees to get 30000 more, for the model it'd still be counted as a bit of overfitting because those are still the same images, with similar pixel densities, which a neural net can easily identify. We thought of scraping more images from flickr using the flickr api to get more training data but decided to move into another direction. Instead of going for the given 8x8 images, we went for the original 75x75 images that could be resized to 32x32 or 28x28 to get essentially more features for the model to play with. For that we wrote a separate script that goes through all images in train folder, rotates each to all 3 normal degrees (90, 180, 270) and return a flattened numpy array of the images. Given we used the pillow (PIL) package for image processing, some images (~16-18) were unable to be processed/reshaped to desired dimensions, so we ignored them. We got out with a numpy ndarray of shape: (40000, 2352) if 28x28x3 sized images or (40000, 3072) if 32x32x3 sized images. This array had a uint8 dtype and we had to normalize the values by dividing by 255.0 after converting the values to float32. We chose the labels as follows: 

```python
# Key is degree, value denotes its chosen label.
train_mapping = {
    "0": 0,
	"90": 1,
	"180": 2,
	"270": 3
}
```


This was while setting up training data. While setting up for testing data, we found out that the way we rotated and labeled our images was different than the way the actual test images were labeled. That is, our 90 degrees was actually 270 degrees in the actual rotated testing data and vice-versa. So, for creating test data, we had to exchange the values of 90 and 270 to get:

```python
test_mapping = {
	"0": 0,
	"90": 3,
	"180": 2,
	"270": 1
}
```


This was the data pre-processing and augmentation step. Before starting training, we added 5 fold cross validation to the model using a little helper function from sklearn. It simply splits the training dataset into 5 random folds: (32000 random training samples + 8000 random validation samples) x 5. We added 2 dense layers with ReLU as the activation function and the final dense layer with 4 nodes (one for each orientation degree) and softmax for classification. Choosing the learning rate for RMSProp was tricky since we didn't want to overshoot with a larger value, so we went with 0.00005 to be safe. Training is done in 1024 mini batches, both forward and backward pass. Iterating through batches was done through a custom k fold class. Each layer (Dense, Activation and Dropout) has its own forward pass and backward pass function. We went with categorical crossentropy as the loss function since it seems to be the standard for classification problems. Upon training, validation was done with accuracy as a metric. The training and validation loss and accuracy is shown for each epoch, given verbose is set to True. Once the training is complete, the train() function returns the best model (fold) to evaluate on the test data. Average test accuracies range from 75-79%.