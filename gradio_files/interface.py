import io
import requests
from PIL import Image
import numpy as np
import gradio as gr
import faiss
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

with open("features/filenames-caltech101.pickle", "rb") as f:
    caltech_filenames = pickle.load(f)
    caltech_filenames = np.array([filename[1:] for filename in caltech_filenames])

with open("features/filenames-voc2012.pickle", "rb") as f:
    voc_filenames = pickle.load(f)
    voc_filenames = np.array([filename[1:] for filename in voc_filenames])

model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="max"
)


def read_online_image(image_url):
    try:  # in case users paste non-image links that cannot be read as image
        response = requests.get(image_url)
        with Image.open(io.BytesIO(response.content)) as img:
            return np.array(img.resize((224, 224)))
    except:
        return None


def find_similar(inp, image_url, data, k):
    index = faiss.read_index(f"gradio/index_ivfpq_{data}.index")
    if image_url:
        inp = read_online_image(image_url)

    if inp is None:
        return 5 * [None]  # fit the output interface

    inp = inp.reshape((-1, 224, 224, 3))
    inp = preprocess_input(inp)
    query = model.predict(inp)

    D, I = index.search(
        query, k + 1
    )  # +1 because don't count the same image itself as neighbor
    result_filenames = (
        caltech_filenames[I[0][1:]] if data == "caltech" else voc_filenames[I[0][1:]]
    )
    result_filenames = result_filenames.tolist() + [None] * (
        5 - len(result_filenames)
    )  # Prevent ValueError: Number of output components does not match number of values returned from from function find_similar

    return result_filenames


inputs = [
    gr.Image(shape=(224, 224), tool="select"),
    gr.Textbox(label="Paste Image Url (If not using query image from collection)"),
    gr.inputs.Dropdown(
        ["caltech", "voc"], label="Choose your Image Collection", default="caltech"
    ),
    gr.Slider(1, 5, value=5, step=1, label="Number of similar images to show"),
]
outputs = [gr.Image() for _ in range(5)]

gr.Interface(
    fn=find_similar,
    inputs=inputs,
    outputs=outputs,
    examples=[
        [caltech_filenames[3810]],
        [caltech_filenames[4938]],
        [voc_filenames[770]],
        [voc_filenames[6730]],
        [voc_filenames[2244]],
    ],  # provides argument to first input in inputs
    #  live=True, # if true only can crop once
    title="Caltech101 and VOC2012 Reverse Image Search",
    description="""
             3 ways to search
                1.Upload your own (can crop)
                2.Paste a link
                3.Choose an example below (can crop)
             
             Tips
                - Use the dropdown to choose the dataset to search in
                - Move the slider to select the number of similar images to search.
                """,
    allow_flagging="never",
).launch(share=True)
