import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

#load the model
model = tf.keras.models.load_model(r"F:\project\handwritten_digit_recognition\my_model.keras")

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#reshape the data to 2D array
X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))

#normalize the dataset
X_train = X_train / 255.0
X_test = X_test / 255.0

# evaluate the model
prediction = model.predict(X_test)
y_pred = np.argmax(prediction, axis=1)
acc = tf.keras.metrics.Accuracy(name='accuracy')
acc.update_state(y_test, y_pred)
print(f"accuracy: {acc.result()}") #accuracy: 0.9659000039100647 -> pretty good

# predictions for 64 random digits
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

def predict(image):
    image_arr = (image['composite'])
    image_arr = 255 - image_arr
    # normalize the arr
    image_arr =  image_arr.astype("float32")/255.0 # (1,784)
    #reshape
    image_arr = image_arr.reshape(1,784)
    #predict
    prediction = model.predict(image_arr) # shape: (10,)
    y_hat = np.argmax(prediction)
    return int(y_hat)

def test(image):
    # global num
    # num = image['composite']
    return image['composite']


iface = gr.Interface(
    fn= predict,
    inputs=gr.Sketchpad(canvas_size=(28,28), type='numpy', image_mode='L', brush=gr.Brush(colors="black", default_size=2)),
    outputs="label",
    title="Digits Recognizer"
)

iface.launch()

