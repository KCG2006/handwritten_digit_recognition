import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

#load the model
model = tf.keras.models.load_model(r"F:\project\handwritten_digit_recognition\my_model.keras")

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# change the value
X_train_bin = np.where(X_train > 0, 255, 0).astype(np.uint8)
X_test_bin = np.where(X_test > 0, 255, 0).astype(np.uint8)


# #reshape the data to 2D array
X_train_bin = X_train_bin.reshape((X_train_bin.shape[0],-1))
X_test_bin = X_test_bin.reshape((X_test_bin.shape[0],-1))

#normalize the dataset
X_train_bin = X_train_bin / 255.0
X_test_bin = X_test_bin / 255.0

# print(X_train[0], y_train[0])
# evaluate the model
# prediction = model.predict(X_test_bin)
# y_pred = np.argmax(prediction, axis=1)
# acc = tf.keras.metrics.Accuracy(name='accuracy')
# acc.update_state(y_test, y_pred)
# print(f"accuracy: {acc.result()}") #accuracy: 0.9659000039100647 -> pretty good

# predictions for 64 random digits
m, n = X_test_bin.shape

fig, axes = plt.subplots(8, 8, figsize=(5, 5))
fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # [left, bottom, right, top]

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Select rows corresponding to the random indices and
    # reshape the image
    X_random_reshaped = X_test_bin[random_index].reshape((28, 28))

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Predict using the Neural Network
    prediction = model.predict(X_test_bin[random_index].reshape(1, 784))
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)

    # Display the label above the image
    ax.set_title(f"{y_test[random_index]},{yhat}", fontsize=10)
    ax.set_axis_off()
fig.suptitle("Label, yhat", fontsize=14)
plt.show()



def preprocessing(image_np):
    """
    image_np: numpy array 28x28, grayscale, giá trị 0-255, đã đảo màu, đã resize
    """
    # Áp bilateral filter giữ nét cạnh
    # blurred = cv2.GaussianBlur(image_np, (5,5), sigmaX=2)
    blurred = cv2.distanceTransform(image_np, cv2.DIST_L2, 5)
    norm = cv2.normalize(blurred, None, 0.0, 1.0, cv2.NORM_MINMAX)

    return norm

def predict(image):
    image_arr = (image['composite'])
    image_arr = 255 - image_arr
    # filter_image_arr = preprocessing(image_arr)
    # normalize the arr
    image_arr =  image_arr.astype("float32")/255.0 # (1,784)
    #reshape
    image_arr = image_arr.reshape(1,784)
    #predict
    prediction = model.predict(image_arr) # shape: (10,)
    y_hat = np.argmax(prediction)
    return int(y_hat)

def test(image):
    image_arr = (image['composite'])
    image_arr = 255 - image_arr
    filter_image_arr = preprocessing(image_arr)
    return filter_image_arr


iface = gr.Interface(
    fn= predict,
    inputs=gr.Sketchpad(canvas_size=(28,28), type='numpy', image_mode='L', brush=gr.Brush(colors="black", default_size=2)),
    outputs=gr.Textbox(),
    title="Digits Recognizer"
)

iface.launch()

