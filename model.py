import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#reshape the data to 2D array
X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))

#normalize the dataset
X_train = X_train / 255.0
X_test = X_test / 255.0

#print out the shape of X_train ,y_train ,X_test, y_test
print(f"X train: {X_train.shape} | y_train: {y_train.shape}")
print(f"X test: {X_test.shape} | y_test: {y_test.shape}")









