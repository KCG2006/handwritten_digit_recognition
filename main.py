import gradio as gr

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
