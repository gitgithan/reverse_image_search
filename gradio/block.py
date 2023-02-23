import io
import os
import random
from functools import partial
import pickle
import numpy as np
import requests
from PIL import Image
import gradio as gr
import faiss
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

global INDEX

with open("features/filenames-caltech101.pickle", "rb") as f:
    caltech_filenames = pickle.load(f)
    caltech_filenames = np.array([filename[1:] for filename in caltech_filenames]) #remove leading / because filenames are created in colab in features folder on root path /features

with open("features/filenames-voc2012.pickle", "rb") as f:
    voc_filenames = pickle.load(f)
    voc_filenames = np.array([filename[1:] for filename in voc_filenames])
    
indexes = {
    "caltech": faiss.read_index("gradio/index_ivfpq_caltech.index"),
    "voc": faiss.read_index("gradio/index_ivfpq_voc.index")
}

model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="max"
)


def read_online_image(image_url):
    try:  # in case users paste non-image links that cannot be read as image
        response = requests.get(image_url)
        return Image.open(io.BytesIO(response.content))
    except:
        return None


def search(inp, data, k):
    global INDEX
    INDEX = indexes[data]  # search index depends on collection selected

    if inp is None:
        return 5 * [None]  # fit the output interface
    inp = np.array(inp.resize((224, 224))).reshape((-1, 224, 224, 3))

    # inp = np.array(inp.resize((224,224))).reshape((-1, 224, 224, 3)) # resize with PIL, then reshape with numpy
    inp = preprocess_input(inp)
    query = model.predict(inp)

    D, I = INDEX.search(
        query, k + 1
    )  # +1 because don't count the same image itself as neighbor
    result_filenames = (
        caltech_filenames[I[0][1:]] if data == "caltech" else voc_filenames[I[0][1:]]
    )
    result_filenames = result_filenames.tolist() + [None] * (
        5 - len(result_filenames)
    )  # Prevent ValueError: Number of output components does not match number of values returned from from function find_similar
    return result_filenames


with gr.Blocks() as demo:
    gr.Markdown(
        """<h1><center>Caltech101 and VOC2012 Reverse Image Search</center></h1>"""
    )
    num_images = 2
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Image to Search")
            with gr.Row():
                with gr.Column():
                    to_search = gr.Image(
                        interactive=True, label="To Search", type="pil"
                    )  # pil type so resize can be called, if no don't specify interactive, Dataset component pointing to this causes this to not allow upload because it's seen as output
                    mirror_curr_prev = gr.Button(
                        value="Copy To Search to Last Searched"
                    )
                with gr.Column():
                    last_search = gr.Image(label="Last Searched", type="pil")
                    mirror_prev_curr = gr.Button(
                        value="Copy Last Searched to To Search"
                    )

                mirror_curr_prev.click(
                    lambda x: x, inputs=to_search, outputs=last_search
                )
                mirror_prev_curr.click(
                    lambda x: x, inputs=last_search, outputs=to_search
                )

            search_btn = gr.Button("Search", variant="primary")
            gr.Markdown("## Examples")

            dataset_choices = {
                "caltech": [
                    [caltech_filenames[i]]
                    for i in np.random.choice(
                        range(len(caltech_filenames)), replace=False, size=50
                    )
                ],
                "voc": [
                    [voc_filenames[i]]
                    for i in np.random.choice(
                        range(len(voc_filenames)), replace=False, size=50
                    )
                ],
            }
            dataset = gr.Dataset(
                components=[to_search],
                samples=[
                    [caltech_filenames[i]]
                    for i in np.random.choice(
                        range(len(caltech_filenames)), replace=False, size=50
                    )
                ],
                samples_per_page=10,
            )

            dataset.click(lambda x: x[0], inputs=dataset, outputs=to_search)

            gr.Markdown("""**Please enter valid image url**""")
            online_image_url = gr.Textbox(
                value="https://cdn.britannica.com/91/181391-050-1DA18304/cat-toes-paw-number-paws-tiger-tabby.jpg",
                placeholder="Link to jpg, png, gif",
            )
            online_image_url.change(
                read_online_image, inputs=online_image_url, outputs=to_search
            )

            collection = gr.Dropdown(
                choices=["caltech", "voc"],
                label="Choose your Image Collection",
                value="caltech",
                interactive=True,
            )

            slider = gr.Slider(
                value=5,
                minimum=1,
                maximum=5,
                step=1,
                interactive=True,
                label="Number of similar images to show",
            )

            gr.Markdown(
                """# Sample of selected image collection
                        click to enlarge, right-click to download"""
            )
            gallery_choices = {
                "caltech": [
                    caltech_filenames[i]
                    for i in np.random.choice(
                        range(len(caltech_filenames)), replace=False, size=50
                    )
                ],
                "voc": [
                    voc_filenames[i]
                    for i in np.random.choice(
                        range(len(voc_filenames)), replace=False, size=50
                    )
                ],
            }
            gallery = gr.Gallery(gallery_choices["caltech"]).style(grid=[5], height=400)
            collection.change(
                lambda collection: gr.Dataset.update(
                    samples=dataset_choices[collection]
                ),
                inputs=collection,
                outputs=dataset,
            )
            collection.change(
                lambda collection: gallery.update(value=gallery_choices[collection]),
                inputs=collection,
                outputs=gallery,
            )

        with gr.Column():
            gr.Markdown("# Similar Images")
            # images interactive False for nicer display where width fits column. If want to edit, use radio buttons at bottom that copies these to To Search box on top left
            similar_images = [
                gr.Image(
                    value=random.choice(caltech_filenames),
                    label=f"Image {i+1}",
                    visible=True,
                    interactive=False,
                )
                for i in range(5)
            ]
            gr.Markdown("## Search with Similar Images")
            search_again = gr.Radio(
                choices=["1", "2", "3", "4", "5"],
                label="Which of the above similar images do you want to search again?",
            )

        def update_search_dropdown(k_neighbors):
            return gr.update(choices=list(range(1, k_neighbors + 1)))

        def mirror_to_preview(
            *idx_and_similar_images,
        ):  # gradio seem to need each input to be passed separately so no concept of "Group of image component" to be passed as one, receiving function does the hard work parsing
            idx = int(idx_and_similar_images[0])
            return idx_and_similar_images[idx]

        slider.change(update_search_dropdown, inputs=slider, outputs=search_again)
        search_again.change(
            mirror_to_preview, inputs=[search_again] + similar_images, outputs=to_search
        )

        def image_visibility(k_neighbors):
            """Gradio cannot create UI components dynamically, so fix first then edit visibility to achieve similar effect"""
            return [gr.update(visible=True) for _ in range(k_neighbors)] + [
                gr.update(visible=False) for _ in range(5 - k_neighbors)
            ]

        slider.change(image_visibility, inputs=slider, outputs=similar_images)

    inputs = [to_search, collection, slider]
    search_btn.click(search, inputs=inputs, outputs=similar_images)
    search_btn.click(lambda x: x, inputs=to_search, outputs=last_search)

if __name__ == "__main__":
    demo.launch(share=True)
