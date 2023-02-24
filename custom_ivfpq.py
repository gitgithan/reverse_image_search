from collections import defaultdict
import heapq
import operator
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances


class CustomIndexIVFPQ:
    BITS2DTYPE = {
        8: np.uint8,
        16: np.uint16,
    }

    def __init__(
        self,
        d: int,
        m: int,
        nlist: int,
        nbits: int,
    ) -> None:
        """Custom IndexIVFPQ implementation.

        Parameters
        ----------
        d
            Dimensionality of the original vectors.
        m
            Number of segments.
        nlist
            Number of coarse centroids for IVF
        nbits
            Number of bits.
        """
        if d % m != 0:
            raise ValueError("d needs to be a multiple of m")

        if nbits not in CustomIndexIVFPQ.BITS2DTYPE:
            raise ValueError(f"Unsupported number of bits {nbits}")

        self.m = m
        self.nlist = nlist
        self.nprobe = 1
        self.k = 2**nbits
        self.d = d
        self.ds = d // m

        self.coarse_quantizer = KMeans(n_clusters=self.nlist, random_state=1)
        self.inverted_list = defaultdict(list)
        self.max_id = 0  # to start following batches of vector adding from the right index to prevent duplicate ids if in different IVF cell or overwriting of data if in same IVF cell
        self.codes_db = np.empty(
            (0, m), dtype=CustomIndexIVFPQ.BITS2DTYPE[nbits]
        )  # always cleared in jupyter for convenience

        self.fine_quantizers = [
            KMeans(n_clusters=self.k, random_state=1) for _ in range(m)
        ]

        self.is_trained = False

        self.dtype = CustomIndexIVFPQ.BITS2DTYPE[nbits]
        self.dtype_orig = np.float32

    def fit_coarse_quantizer(self, feature_list):
        """Coarse quantizer to divide database vectors into various voronoi cells so search only probes nprobe closest

        Parameters
        ----------
        feature_list
            Data to be converted to residuals

        Returns
        -------
        feature_list_residual
            Residuals created by subtracting each vector with it's own coarse centroid
        """
        self.coarse_quantizer.fit(feature_list)
        feature_list_residual = (
            feature_list
            - self.coarse_quantizer.cluster_centers_[self.coarse_quantizer.labels_]
        )  # generate residual database vectors to be fine quantized
        return feature_list_residual

    def fit_fine_quantizer(self, feature_list_residual):
        """Fit m fine quantizers for each of m segments

        Parameters
        ----------
        feature_list_residual
            Residuals created by subtracting each vector with it's own coarse centroid
        """
        for i in range(self.m):
            X_i = feature_list_residual[:, i * self.ds : (i + 1) * self.ds]
            self.fine_quantizers[i].fit(X_i)

    def apply_coarse_quantizer(self, feature_list):
        """Find closest IVF centroid using coarse quantizer and get residuals

        Parameters
        ----------
        feature_list
            Raw Vectors to be assigned to coarse quantizer centroids

        Returns
        -------
        feature_list_residual
            Residuals to be quantized by fine quantizer after coarse quantization

        ivf_cells
            labels of which coarse centroids each vector goes into
        """
        ivf_cells = self.coarse_quantizer.predict(feature_list)
        feature_list_residual = (
            feature_list - self.coarse_quantizer.cluster_centers_[ivf_cells]
        )  # generate residual database vectors to be fine quantized
        return feature_list_residual, ivf_cells

    def quantize_residuals(self, feature_list_residual):
        """
        Fine quantization of residuals of both database and query vectors

        Parameters
        ----------
        feature_list_residual
            Residuals created by subtracting each vector with it's own coarse centroid

        Returns
        -------
        codes
            Quantized codes of residuals, shaped n, m
        """
        n = len(feature_list_residual)
        codes = np.empty(
            (n, self.m), dtype=self.dtype
        )  # Prevents automatic 'float64' causing IndexError: arrays used as indices must be of integer (or boolean) type

        for i in range(self.m):
            estimator = self.fine_quantizers[i]
            X_i = feature_list_residual[:, i * self.ds : (i + 1) * self.ds]
            codes[:, i] = estimator.predict(
                X_i
            )  # shape n number of vectors, m segments

        return codes

    def add_inverted_list(self, ivf_cells, codes):
        """
        Assign ids to cells and add codes to database

        Parameters
        ----------
        ivf_cells
            coarse quantized labels of vectors
        codes
            Quantized codes of residuals to be added to database of quantized vectors
        """
        for idx, coarse_center in enumerate(ivf_cells, start=self.max_id):
            self.inverted_list[coarse_center].append(idx)

        self.max_id += len(
            codes
        )  # update max_id so next addition to IVF don't duplicate id (if same different coarse_center) or overwrite data (if same coarse_center)
        self.codes_db = np.vstack([self.codes_db, codes])

    def distance_to_IVFcentroids(self, query):
        """
        Find distance of raw query to coarse centroids

        Parameters
        ----------
        query
            Raw query vector

        """
        query_distance_to_coarse_centroids = euclidean_distances(
            query, self.coarse_quantizer.cluster_centers_, squared=True
        )[0]
        nearest_inverted_keys = np.argsort(query_distance_to_coarse_centroids)[
            : self.nprobe
        ]  # argsort gives index of closest coarse centroids, to be filtered by nprobe
        return nearest_inverted_keys

    def generate_query_residual(self, query, current_cell):
        """Find closest IVF centroid and return residual of query from that centroid

        Parameters
        ----------
        query
            Raw query vector
        current_cell
            A particular coarse centroid explored during probing for IVF cells

        Returns
        -------
        """
        query_residual = (
            query - self.coarse_quantizer.cluster_centers_[current_cell]
        )  # generate residual query to be compared against all quantized residuals

        return query_residual

    def compute_distance_table(self, query_residual):
        """Distance table per coarse centroid for reuse by all quantized residual vectors in same cell

        Parameters
        ----------
        query_residual
            Residual of query vector

        Returns
        -------
        distance_table
            Table of distances from each query vector to all k clusters, for each segment, to be used by all database vectors in same coarse centroid
        """
        distance_table = np.empty(
            (self.m, self.k), dtype=self.dtype_orig
        )  # shape m segments, distance to k clusters

        for i in range(self.m):
            X_i = query_residual[:, i * self.ds : (i + 1) * self.ds]
            centers = self.fine_quantizers[i].cluster_centers_  # (k, ds)
            distance_table[i, :] = euclidean_distances(X_i, centers, squared=True)
        return distance_table

    def filter_residual_vectors(self, current_cell):
        """Identify only relevant vectors in same cell as query to compute distances for

        Parameters
        ----------
        current_cell
            particular coarse centroid explored during probing for IVF cells

        Returns
        -------
        filtered_result
            Filtered codes so search only calculates their distances
        filtered_ids
            Filtered ids used for indicating which vectors are being searched
        """

        filtered_ids = self.inverted_list[current_cell]
        filtered_result = self.codes_db[filtered_ids]

        return filtered_result, filtered_ids

    def calculate_distances(self, filtered_result, distance_table):
        """Calculate distance of each database vector with quantized query vector

        Parameters
        ----------
        filtered_result
            Filtered codes for calculating distances with their coarse centroid
        distance_table
            Table of distances from each query vector to all k clusters, for each segment, to be used by all database vectors in same coarse centroid

        Returns
        -------
        distances
            Distance of each database vector with quantized query vector
        """
        distances = np.zeros(len(filtered_result), dtype=self.dtype_orig)

        for i in range(m):
            distances += distance_table[i, filtered_result[:, i]]

        return distances

    def find_smallest_k(self, distances, filtered_ids, k_nearest):
        """Find nearest k neighbors (including self)

        Parameters
        ----------
        distances
            Distance of each database vector with quantized query vector
        filtered_ids
            ids of vectors
        k_nearest
            Number of approximate neighbors


        Returns
        -------
        D
            K nearest distances per query vector
        I
            Indices of the K nearest distances per query vector
        """
        distance_id = zip(distances, filtered_ids)
        D, I = zip(*heapq.nsmallest(k_nearest, distance_id, operator.itemgetter(0)))

        return D, I

    def train(self, feature_list: np.ndarray) -> None:
        """Train the index given data

        Parameters
        ----------
        feature_list
            Array of shape `(n, d)` and dtype `float32`.
        """
        feature_list_residual = self.fit_coarse_quantizer(feature_list)
        self.fit_fine_quantizer(feature_list_residual)

        self.is_trained = True

    def add(self, feature_list: np.ndarray) -> None:
        """Add vectors to the database (their encoded versions).

        Parameters
        ----------
        feature_list
            Array of shape `(n_codes, d)` of dtype `np.float32`.
        Raises
        ------
        ValueError
            Cannot add data if quantizers not trained
        """
        if not self.is_trained:
            raise ValueError(
                "Both coarse and fine quantizers need to be trained first."
            )

        feature_list_residual, ivf_cells = self.apply_coarse_quantizer(feature_list)
        codes = self.quantize_residuals(feature_list_residual)
        self.add_inverted_list(ivf_cells, codes)

    def search(self, query: np.ndarray, k_nearest: int) -> tuple:
        """Search for k nearest neighbors

        Parameters
        ----------
        query
            Raw query vector
        k_nearest
            Number of approximate neighbors


        Returns
        -------
        D
            K nearest distances per query vector
        I
            Indices of the K nearest distances per query vector

        Raises
        ------
        ValueError
            Cannot add data if quantizers not trained
            Cannot search database if it's empty

        """
        if not self.is_trained:
            raise ValueError(
                "Both coarse and fine quantizers need to be trained first."
            )

        if self.codes_db.size == 0:
            raise ValueError("No codes detected. You need to run `add` first")

        nearest_inverted_keys = self.distance_to_IVFcentroids(query)

        nprobe_distances = np.array([], dtype=self.dtype_orig)
        nprobe_filtered_ids = np.array([], dtype=np.uint64)

        for current_cell in nearest_inverted_keys:
            query_residual = self.generate_query_residual(query, current_cell)
            distance_table = self.compute_distance_table(query_residual)
            filtered_result, filtered_ids = self.filter_residual_vectors(current_cell)
            distances = self.calculate_distances(filtered_result, distance_table)

            nprobe_distances = np.append(nprobe_distances, distances)
            nprobe_filtered_ids = np.append(nprobe_filtered_ids, filtered_ids)

        D, I = self.find_smallest_k(nprobe_distances, nprobe_filtered_ids, k_nearest)

        return D, I


from sklearn.decomposition import PCA

if __name__ == "__main__":
    d = 128
    m = 8
    nlist = 100
    nbits = 8
    custom_ivfpq = CustomIndexIVFPQ(d, m, nlist, nbits)

    feature_list = pickle.load(open("features/features-caltech101-resnet.pickle", "rb"))
    # feature_list = feature_list/np.linalg.norm(feature_list,axis=1).reshape(-1,1)  # normalize features so each column has length 1
    pca = PCA(n_components=128)
    feature_list_compressed = pca.fit_transform(feature_list)

    custom_ivfpq.train(feature_list_compressed)
    custom_ivfpq.add(feature_list_compressed)

    k_nearest = 6

    nprobe_test = {}
    # nprobes_to_test = range(1,4)
    nprobes_to_test = [1]

    for nprobe in nprobes_to_test:
        D_list = []
        I_list = []

        for query in feature_list_compressed:
            D, I = custom_ivfpq.search(
                query.reshape(1, -1), k_nearest  # sklearn euclidean_distances needs 2D
            )
            D_list.append(D)
            I_list.append(I)

        D_list = np.array(D_list, dtype=np.float32)
        I_list = np.array(I_list, dtype=np.uint64)

        nprobe_test[nprobe] = (D_list, I_list)

    print("I_list: ", I_list.shape)
    print("Sample Indices of first 5 query points: ", I_list[:5])
