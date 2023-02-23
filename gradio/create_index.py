import argparse
import faiss
import pickle
from sklearn.decomposition import PCA


def create_index(data):
    datasets = {
        "caltech": "features/features-caltech101-resnet.pickle",
        "voc": "features/features-voc2012-resnet.pickle",
    }

    feature_list = pickle.load(open(datasets[data], "rb"))
    # feature_list = feature_list/np.linalg.norm(feature_list,axis=1).reshape(-1,1)  # normalize features so each column has length 1

    index_ivfpq = faiss.index_factory(2048, "IVF100,PQ8")
    index_ivfpq.train(feature_list)
    index_ivfpq.add(feature_list)

    faiss.write_index(index_ivfpq, f"gradio/index_ivfpq_{data}.index")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", choices=["caltech", "voc"], help="Select your dataset"
    )

    args = parser.parse_args()
    create_index(args.data)
