import gradio as gr

def sketchToNumpy(image):
    imArray = (image['composite'])
    return imArray.shape

iface = gr.Interface(
    fn=sketchToNumpy,
    inputs=gr.Sketchpad(crop_size=(28,28), type='numpy', image_mode='L', brush=gr.Brush()),
    outputs=gr.Textbox(),
    title="Digits Recognizer"
)

iface.launch()
