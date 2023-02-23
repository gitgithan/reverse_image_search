# Setup :wrench:

1. **Create virtual environment and install requirements**

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Download data** (required if want to run gradio and show images)
   - `chmod u+x data_download.sh` to make script executable
   - `./data_download all` to download both caltech (1 minute) and voc2012 (6 minutes :warning:) (`all` can be substituted with `caltech` or `voc`)
3. **Create Indexes** (optional since already created)
   - `python gradio/create_index.py --data caltech` (`caltech` can be substituted with `voc`)
4. **Run gradio** (ensure `features` and `datasets` folder exist at same level as gradio folder)
   - `python gradio/block.py`

# Folder Structure :file_folder:

```
.
├── Makefile
├── README.md
├── __pycache__
│   ├── custom_ivfpq.cpython-38.pyc
│   └── test_custom_ivfpq.cpython-38-pytest-7.2.1.pyc
├── custom_ivfpq.py
├── custom_ivfpq_faiss.py
├── data_download.sh
├── datasets
│   ├── VOCdevkit
│   └── caltech101
├── features
│   ├── class_ids-caltech101.pickle
│   ├── features-caltech101-resnet-finetuned.pickle
│   ├── features-caltech101-resnet.pickle
│   ├── features-caltech101-resnetscratch.pickle
│   ├── features-voc2012-resnet.pickle
│   ├── filenames-caltech101.pickle
│   └── filenames-voc2012.pickle
├── gradio
│   ├── block.py
│   ├── create_index.py
│   ├── index_ivfpq_caltech.index
│   ├── index_ivfpq_voc.index
│   └── interface.py
├── ivfpq.pptx
├── notebooks
│   ├── feature_extraction.ipynb
│   ├── index_search.ipynb
│   ├── runtime_experiments.ipynb
│   └── visualizations.ipynb
├── requirements.txt
└── test_custom_ivfpq.py
```

- `features` (366.5MB) and `datasets` (2GB) are not commited to version control
  - `features` - train your own using `feature_extraction.ipynb`, then download from colab to local
  - `datasets` - download using `data_download.sh` or manually

# File Content :books:

1. `custom_ivfpq_faiss.py` - pure python (except clustering section) implementation of [IVFPQ paper](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf), with tweaks in inverted file data structure
   - run with `python custom_ivfpq_faiss.py`, < 2 seconds to train
   - Development process in `notebooks/index_search.ipynb`
   - Summary of API design in `ivfpq.pptx`
2. `custom_ivfpq.py` - same content as previous, except using sklearn Kmeans instead of faiss.Kmeans
   - run with `python custom_ivfpq.py`, 40 seconds to fit, 20 seconds to predict all ~9000 caltech101 images
3. `gradio`
   - `block.py` - blocks (low level) gradio api, allows complete control of data flow
   - `interface.py` - interface (high level) gradio api, limited control of data flow
   - `create_index.py` - script to create indexes to store in same gradio folder, to be loaded by gradio for search
4. `notebooks`

   - `feature_extraction.ipynb` - Convert data to dense feature by fine-tuning models, to be indexed for search
     - run in colab for GPU, filepaths generated there are on root `/features`, `/datasets` to prevent network transfer latency with google drive, should remove leading / if run locally to prevent messing with filesystem
   - `visualizations.ipynb` - [Federpy](https://github.com/zilliztech/feder) visualizations of faiss IndexIVFFlat and hnswlib

     - run in colab because hnswlib cannot be installed locally (`ERROR: Could not build wheels for hnswlib which use PEP 517 and cannot be installed directly`)

   - `runtime_experiments.ipynb`- Experiments on index/search runtime performances of sklearn KNN and Annoy libraries
     - :warning: Start notebook with `%chdir ..` so current directory contains `notebooks` and `open()` can access `features`, `datasets`
   - `index_search.ipynb` - Experiments on speed, memory and recall tradeoffs of faiss ANN algorithms and custom implementations of IVFPQ (inverted file index with product quantization)
     - :warning: Start notebook with `%chdir ..` so current directory contains `notebooks` and `open()` can access `features`, `datasets`
