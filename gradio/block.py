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
from cv2 import imencode
import base64

global INDEX

# Use faster converter, maybe not needed in gradio 4.0 : https://github.com/gradio-app/gradio/issues/2635#issuecomment-1423531319
def encode_pil_to_base64_new(pil_image):
    print("using new encoding method")
    image_arr = np.asarray(pil_image)[:,:,::-1]
    _, byte_data = imencode('.png', image_arr)        
    base64_data = base64.b64encode(byte_data)
    base64_string_opencv = base64_data.decode("utf-8")
    return "data:image/png;base64," + base64_string_opencv

gr.processing_utils.encode_pil_to_base64 = encode_pil_to_base64_new

with open("features/filenames-caltech101.pickle", "rb") as f:
    caltech_filenames = pickle.load(f)
    caltech_filenames = np.array(
        [filename[1:] for filename in caltech_filenames]
    )  # remove leading / because filenames are created in colab in features folder on root path /features

with open("features/filenames-voc2012.pickle", "rb") as f:
    voc_filenames = pickle.load(f)
    voc_filenames = np.array([filename[1:] for filename in voc_filenames])

indexes = {
    "caltech": faiss.read_index("gradio/index_ivfpq_caltech.index"),
    "voc": faiss.read_index("gradio/index_ivfpq_voc.index"),
}

dataset_choices = {
    "caltech": 
        [
        [caltech_filenames[i]]
        for i in np.random.choice(
            range(len(caltech_filenames)), replace=False, size=20
        )
    ],
    "voc": 
        [
        [voc_filenames[i]]
        for i in np.random.choice(
            range(len(voc_filenames)), replace=False, size=20
        )
    ],
}

def encode_pil_to_base64_new(pil_image):
    print("using new encoding method")
    image_arr = np.asarray(pil_image)[:,:,::-1]
    _, byte_data = imencode('.png', image_arr)        
    base64_data = base64.b64encode(byte_data)
    base64_string_opencv = base64_data.decode("utf-8")
    return "data:image/png;base64," + base64_string_opencv

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

def load_samples(file_idx):
    # State management pattern at https://github.com/gradio-app/gradio/issues/3271#issuecomment-1440455811
    global samples
    return samples[file_idx][0] # extract first element because Dataset constructed with list of list by gradio api design

def update_examples(collection_value):
    global samples
    samples = dataset_choices[collection_value]
    return gr.Dataset.update(samples=samples)

def mirror_to_preview(
    *idx_and_similar_images,
):  # gradio seem to need each input to be passed separately so no concept of "Group of image component" to be passed as one, receiving function does the hard work parsing
    idx = int(idx_and_similar_images[0])
    return idx_and_similar_images[idx]

def update_search_dropdown(k_neighbors):
    return gr.update(choices=list(range(1, k_neighbors + 1)))


def image_visibility(k_neighbors):
    """Gradio cannot create UI components dynamically, so fix first then edit visibility to achieve similar effect"""
    return [gr.update(visible=True) for _ in range(k_neighbors)] + [
        gr.update(visible=False) for _ in range(5 - k_neighbors)
    ]
    
online_image_url = gr.Textbox(
    placeholder="Link to jpg, png, gif"
)  # https://gradio.app/controlling-layout/#defining-and-rendering-components-separately

################### Explaining css ###################

# overflow-y: scroll !important; because without !important it is being overwritten by parent container style of overflow: visible

similar_css = """#similar {
                            height: 950px;
                            overflow-y: scroll !important; 
                          }
                 h2 span { font-size:16px; }
                 h1, h2 { margin: 0 !important; }
                 .gradio-container { background-image: url('file=figures/unicorn.png');
                                     background-size: contain;
                                     max-width: 80% !important;
                                     
                        }
                        
                div#gallery_search > div:nth-child(3) {
                                     min-height: 700px;
                            }
         """


samples = dataset_choices['caltech'] # to define it for dataset.click(load_samples

model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="max"
)


with gr.Blocks(css=similar_css) as demo:
    gr.Markdown(
        """<h1><center>Caltech101 and VOC2012 Reverse Image Search</center></h1>"""
    )

    with gr.Row():
        with gr.Column():
            with gr.Row():
                gr.Markdown(
                    "<h2>Image to Search 🔍</h2>"
                )
                upload_webcam = gr.Checkbox(label="Check box to use Webcam")
                
            with gr.Row():
                with gr.Column():
                    to_search_webcam = gr.Image(
                        interactive=True,
                        label="To Search 🔮",
                        source="webcam",
                        visible=False,
                        mirror_webcam=False,  # prevent weird UX when users mirroring to and fro between To Search and Last Searched
                        type="pil",
                    )  # pil type so resize can be called, if no don't specify interactive, Dataset component pointing to this causes this to not allow upload because it's seen as output

                    to_search = gr.Image(
                        interactive=True,
                        label="To Search 🔮",
                        source="upload",
                        visible=True,
                        mirror_webcam=False,  # prevent weird UX when users mirroring to and fro between To Search and Last Searched
                        type="pil",
                    )  # pil type so resize can be called, if no don't specify interactive, Dataset component pointing to this causes this to not allow upload because it's seen as output
                    mirror_curr_prev = gr.Button(
                        value="Copy To Search to Last Searched ➡️"
                    )
                    
                    upload_webcam.change(
                        lambda checked: gr.update(visible=checked),
                        inputs=upload_webcam,
                        outputs=to_search_webcam,
                    )

                with gr.Column():
                    last_search = gr.Image(label="Last Searched ✅", type="pil")
                    mirror_prev_curr = gr.Button(
                        value="⬅️ Copy Last Searched to To Search"
                    )
                to_search_webcam.change(
                    lambda x: x, inputs=to_search_webcam, outputs=to_search
                )
                mirror_curr_prev.click(
                    lambda x: x, inputs=to_search, outputs=last_search
                )
                mirror_prev_curr.click(
                    lambda x: x, inputs=last_search, outputs=to_search
                )
                
            with gr.Row():
                gr.Markdown(
                    "<h2>Choose Collection 🎨</h2>",
                )
                collection = gr.Radio(
                    choices=["caltech", "voc"],
                    value="caltech",
                    interactive=True,
                    label="",
                )
                
            search_btn = gr.Button("Search", variant="primary")
            with gr.Tab("Use examples from collection"):
                gr.Markdown("**Click any example to populate To Search 💡**")
                
                dataset = gr.Dataset(
                    components=[to_search],
                    samples=samples, # default start with caltech samples defined in global scope, don't write dataset_choices['caltech'] or else samples undefined in dataset.click(load_samples)
                    samples_per_page=20,
                    type="index"
                )
                
                dataset.click(load_samples, inputs=dataset, outputs=to_search)

            with gr.Tab("Use public image url"):
                gr.Markdown("""**Please click examples or enter valid image url** 📭""")
                gr.Examples(
                    [
                        [
                            "https://nationaltoday.com/wp-content/uploads/2020/04/unicorn-1-1.jpg"
                        ],
                        [
                            "https://stpaulpet.com/wp-content/uploads/dog-facts-cat-facts.jpg"
                        ],
                    ],
                    online_image_url,
                )
                online_image_url.render()
                online_image_url.change(
                    read_online_image, inputs=online_image_url, outputs=to_search
                )

            gr.Markdown(
                """<h2>Gallery   <span>Click to enlarge, Right-click to download</span></h2>"""
            )
            gallery_choices = {
                "caltech":[
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
            gallery = gr.Gallery(gallery_choices["caltech"],elem_id="gallery_search").style(grid=[4])
            collection.change(
                lambda collection: gallery.update(value=gallery_choices[collection]),
                inputs=collection,
                outputs=gallery,
            )
            
            collection.change(
                update_examples,
                inputs=collection,
                outputs=dataset,
            )

        with gr.Column():
            gr.Markdown("<h2><center>Similar Images 🔎</center></h2>")
            initial_slider_value = 3
            with gr.Row():
                slider = gr.Slider(
                    value=initial_slider_value,
                    minimum=1,
                    maximum=5,
                    step=1,
                    interactive=True,
                    label="Number of similar images to show",
                )
                search_again = gr.Radio(
                    choices=list(map(str, range(1, initial_slider_value + 1))),
                    label="Search output image? ♻️",
                )
            with gr.Box(
                elem_id="similar"
            ):  # need this wrapper to group images together for css targeting, and to separate from Radio component below
                similar_images = [
                    gr.Image(
                        value=random.choice(caltech_filenames),
                        label=f"Image {i+1}",
                        visible=i < initial_slider_value,
                        interactive=False,  # images interactive False for nicer display where width fits column. If want to edit, use radio buttons at bottom that copies these to To Search box on top left
                    )
                    for i in range(5)
                ]

            slider.change(update_search_dropdown, inputs=slider, outputs=search_again)
            slider.change(image_visibility, inputs=slider, outputs=similar_images)
            search_again.change(
                mirror_to_preview, inputs=[search_again] + similar_images, outputs=to_search
            )


    inputs = [to_search, collection, slider]
    search_btn.click(search, inputs=inputs, outputs=similar_images)
    search_btn.click(lambda x: x, inputs=to_search, outputs=last_search)
    collection.change(search, inputs=inputs, outputs=similar_images)

if __name__ == "__main__":
    demo.launch(share=True, 
                debug=True, 
                server_name="0.0.0.0",
                ssl_keyfile="key.pem", ssl_certfile="cert.pem"
                )
