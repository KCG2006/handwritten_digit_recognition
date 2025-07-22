import gradio as gr
import tensorflow as tf
import os

#load the model
model = tf.keras.models.load_model(r"F:\project\handwritten_digit_recognition\my_model.keras")


def predict(image):
    imArr = (image['composite'])
    return imArr.shape

iface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(crop_size=(28,28), type='numpy', image_mode='L', brush=gr.Brush()),
    outputs=gr.Textbox(),
    title="Digits Recognizer"
)

iface.launch()
