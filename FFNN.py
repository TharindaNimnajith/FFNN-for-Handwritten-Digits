
## Recognizing Handwritten Digits using a Neural Network


## Features : Pixels of the image
## Labels   : 0 - 9 digits

## TensorFlow : An end-to-end  free and open-source machine learning software library platform
## Keras      : An open-source neural-network library for Python which is capable of running on top of TensorFlow

## MNIST Database : A large database of handwritten digits


from keras.datasets import mnist


(train_data, train_target), (test_data, test_target) = mnist.load_data()


print(train_data.shape)
## (60000, 28, 28)
## features - 3d array with 60000 of 28x28 images (for training)

print(train_target.shape)
## (60000,)
## targets - 60000 labels (for training)

print(test_data.shape)
## (10000, 28, 28)
## features - 3d array with 10000 of 28x28 images (for testing)

print(test_target.shape)
## (10000,)
## targets - 10000 labels (for testing)


from matplotlib import pyplot as plt


print(train_target[0])
## 5

plt.imshow(train_data[0])
## image of 5 in RGB


print(train_target[1])
## 0

plt.imshow(train_data[1], cmap = 'gray')
## image of 0 in gray scale
## digit in white color with black background
## advantage: most of the image has 0 for pixel values saving storage


from keras.models import Sequential


model = Sequential()
## an empty NN


## Sequential NN type is used to implement the NN layer by layer sequentially
## The last layer to be added will be the Output Layer

## Don't add Input Layer to the NN
## First layer to add to the NN is the first Hidden Layer


from keras.layers import Flatten, Dense


model.add(Flatten(input_shape = (28, 28)))
## Can't add 2d arrays (images) to FFNN
## Can add 2d arrays to Convolutional NN
## Flatten - Flattening an image
## Converting 2d array to 1d array
## 28x28 is converted to 1x784


model.add(Dense(512, activation = 'relu'))
## 1st hidden layer
## using ReLU activation function

model.add(Dense(100, activation = 'relu'))
## 2nd hidden layer
## using ReLU activation function


## Adding more hidden layers doesn't going to increase accuracy


model.add(Dense(10, activation = 'softmax'))
## last layer
## using Softmax activation function


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
## using the Adam optimizer
## As this is a classification problem using the Categorical Cross Entropy loss function
## metrics = ['accuracy'] shows the accuracy while training
## otherwise only the loss will be shown


## Need to scale down the data to enter data to NN
## All pixel values can be converted to 0-1 range by dividing by 255


train_data = train_data / 255.0
## train_data -> 60000x28x28 array

print(train_data[:5])


test_data = test_data / 255.0
## test_data -> 10000x28x28 array

print(test_data[:5])


print(train_target[:5])
## [5 0 4 1 9]
## can't add to NN in this format in classifications


from keras.utils import to_categorical


train_target = to_categorical(train_target)
## To convert to the correct format to add to the NN

print(train_target[:5])
## [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
##  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
##  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
##  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
##  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]


## Training the model


model.fit(train_data, train_target, epochs = 5)
## epoch - going through the same whole dataset once
## loss decreases in each epoch
## accuracy increases in each epoch


## Predicting using NN


plt.imshow(test_data[0])
## Image of 7


result = model.predict([[test_data[0]]])
## predicting

print(result)


result = result.round()
## rounding the result

print(result)
## [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
## 7 is the 1, others are 0
