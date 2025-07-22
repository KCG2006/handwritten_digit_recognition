import tensorflow as tf

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = X_train / 255.0
x_test = X_test / 255.0

