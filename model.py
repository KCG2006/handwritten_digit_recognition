import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# change the value
X_train_bin = np.where(X_train > 0, 255, 0).astype(np.uint8)
X_test_bin = np.where(X_test > 0, 255, 0).astype(np.uint8)

#reshape the data to 2D array
X_train_bin = X_train_bin.reshape((X_train_bin.shape[0],-1))
X_test_bin = X_test_bin.reshape((X_test_bin.shape[0],-1))

#normalize the dataset
X_train_bin = X_train_bin / 255.0
X_test_bin = X_test_bin / 255.0

#print out the shape of X_train ,y_train ,X_test, y_test
print(f"X train: {X_train_bin.shape} | y_train: {y_train.shape}")
print(f"X test: {X_test_bin.shape} | y_test: {y_test.shape}")

#plot random samples
# m, n = X_train.shape
#
# fig, axes = plt.subplots(8, 8, figsize=(5, 5))
# fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]
#
# for i, ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)
#
#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X_train[random_index].reshape((28,28))
#
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')
#
#     # Display the label above the image
#     ax.set_title(y_train[random_index])
#     ax.set_axis_off()
#     fig.suptitle("Label, image", fontsize=14)
# plt.show()

#define the structure of the model
model = Sequential([
    tf.keras.Input(shape=(784,)),
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear'),
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)
model.fit(X_train_bin, y_train, epochs=40)

# predictions for 64 random digits
# m, n = X_train.shape

# fig, axes = plt.subplots(8, 8, figsize=(5, 5))
# fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

# for i, ax in enumerate(axes.flat):
#     # Select random indices
#     random_index = np.random.randint(m)

#     # Select rows corresponding to the random indices and
#     # reshape the image
#     X_random_reshaped = X_train[random_index].reshape((28, 28))
#
#     # Display the image
#     ax.imshow(X_random_reshaped, cmap='gray')
#
#     # Predict using the Neural Network
#     prediction = model.predict(X_train[random_index].reshape(1, 784))
#     prediction_p = tf.nn.softmax(prediction)
#     yhat = np.argmax(prediction_p)
#
#     # Display the label above the image
#     ax.set_title(f"{y_train[random_index]},{yhat}", fontsize=10)
#     ax.set_axis_off()
# fig.suptitle("Label, yhat", fontsize=14)
# plt.show()
#save model
model.save('my_model.keras')

# evaluate the model
prediction = model.predict(X_test_bin)
y_pred = np.argmax(prediction, axis=1)
acc = tf.keras.metrics.Accuracy(name='accuracy')
acc.update_state(y_test, y_pred)
print(f"accuracy: {acc.result()}") #accuracy: 0.9659000039100647 -> pretty good