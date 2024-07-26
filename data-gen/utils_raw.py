from dataclasses import dataclass
import numpy as np
import random
import h5py
import os
from tqdm import tqdm
import numba
from loguru import logger
import sys

log_file_path = 'logs.log'  # Change this to your desired log file path

# Configure logger to write to both console and file
logger.remove()  # Remove any previous handlers (optional)
logger.add(sys.stdout, colorize=True,
           format="<green>{time}</green> <level>{message}</level>")
logger.add(log_file_path, rotation="10 MB", compression="zip",
           format="{time:YYYY-MM-DD HH:mm:ss} {level} - {message}")


@numba.jit(
    nopython=True,
    # parallel=True,

)
def get_all_masks(all_vector_a):
    num_vectors = all_vector_a.shape[0]
    result = []
    for i in numba.prange(num_vectors):
        vector_a = all_vector_a[i]
        vector_a_mask = np.argwhere(vector_a == 1)[0][0]
        result.append(vector_a_mask)
    return np.array(result)


@numba.jit(
    nopython=True,
    parallel=True,

)
def dot_product_batch(all_vector_a, list_of_vectors):
    """
    Compute the dot product between a batch of vectors and a reference vector.

    Parameters:
        all_vector_a (ndarray): The reference vector of shape (N,).
        list_of_vectors (ndarray): The batch of vectors of shape (M, N).

    Returns:
        ndarray: The dot product between each mask and each vector in the batch.
            The resulting array has shape (M, K), where K is the number of masks.

    Notes:
        - This function uses numba.jit for just-in-time compilation, which improves performance.
        - The function supports parallel execution using the parallel=True argument.
    """
    all_masks = get_all_masks(all_vector_a)
    num_vectors = list_of_vectors.shape[0]
    num_masks = all_masks.shape[0]
    result = np.zeros((num_masks, num_vectors), dtype=np.uint32)
    for i in numba.prange(num_vectors):
        vector_b = list_of_vectors[i]
        for j in numba.prange(num_masks):
            mask = all_masks[j]
            value = 1*(vector_b[mask] == 1)
            if value > 0:
                result[j][i] = value

    return result


@numba.njit(parallel=True)
def masked_sort(test_distance, valid_train, top_k):
    """
    This function takes in three parameters: `test_distance`, `valid_train`, and `top_k`.
    `test_distance` is a numpy array representing the distance between test vectors and training vectors.
    `valid_train` is a numpy array representing the valid training vectors.
    `top_k` is an integer representing the number of top indices to return.

    The function first determines the shape of `test_distance` and `valid_train` arrays.
    It then creates an empty numpy array `result` with shape (n_vector, top_k) and dtype=np.int64.

    Next, it iterates over each vector in `test_distance` using parallel processing.
    For each vector, it finds the indices of valid entries using a mask.
    It then sorts the indices based on distances and takes the top_k indices.
    Finally, it assigns the top_k indices to the corresponding row in the `result` array.

    The function returns the `result` array.
    """
    n_vector, m_distance = test_distance.shape
    # n_vector, m_0_1 = valid_train.shape

    result = np.empty((n_vector, top_k), dtype=np.int64)

    for i in numba.prange(n_vector):
        # Find indices of valid entries using the mask
        valid_indices = np.where(valid_train[i] == 1)[0]

        # Sort indices based on distances
        sorted_indices = np.argsort(test_distance[i, valid_indices])
        # if i < 3:
        #     print(i,sorted_indices.shape,valid_indices.shape)
        #     print(sorted_indices)
        #     print(np.sort(test_distance[i, valid_indices]))
        #     print(valid_indices[sorted_indices[:10]][:10])
        # Take the top_k indices
        result[i, :] = valid_indices[sorted_indices[:top_k]]

    return result


def vectorized_cosine_distances(train, test):
    """
        Calculate the cosine distances between two sets of vectors.

        Parameters:
            train (numpy.ndarray): An array of shape (n_train, d) representing the training set of vectors, where n_train is the number of training samples and d is the dimension of each vector.
            test (numpy.ndarray): An array of shape (n_test, d) representing the test set of vectors, where n_test is the number of test samples and d is the dimension of each vector.

        Returns:
            numpy.ndarray: An array of shape (n_train, n_test) containing the cosine distances between each pair of vectors in the training and test sets.

        Raises:
            ValueError: If the shape of `train` or `test` is invalid.

        Notes:
            - The cosine distance is calculated as 1 - the cosine similarity between two vectors.
            - If the shape of `train` or `test` is invalid, a `ValueError` is raised.
    """
    try:
        # Normalize vectors
        norm_train = np.linalg.norm(train, axis=1, keepdims=True)
        norm_test = np.linalg.norm(test, axis=1, keepdims=True)

        # Compute dot products
        dot_products = np.dot(train, test.T)

        # Compute cosine distances
        distances = dot_products / (norm_train * norm_test.T)

        return 1 - distances
    except ValueError:
        return None


def vectorized_eucliden_distance(train, test):
    # Calculate squared differences element-wise
    train_squared = np.sum(train**2, axis=1, keepdims=True)
    test_squared = np.sum(test**2, axis=1, keepdims=True)
    inner_product = np.dot(train, test.T)
    distances = np.sqrt(train_squared - 2 * inner_product + test_squared.T)
    return distances


@dataclass
class FANNConfig:
    data_source_path: str
    data_sink_path: str
    dataset_name: str
    distance_type: str
    train_batch: list
    selectivity_range: list


class FannDataLoader:
    def __init__(self, config: FANNConfig):
        self.config = config
        self.files_to_load = []
        pass

    def get_raw_data(self):
        """
        Returns the raw train and test data along with the similarity metric used.

        Parameters:
            None

        Returns:
            train_vectors (ndarray): An array containing the training vectors.
            test_vectors (ndarray): An array containing the test vectors.
            distance_type (str): The distance used.
        Note:
            This function must be implemented by the subclass.
        """
        raise NotImplementedError

    def normalize_data(self):
        """
        take data from config.data_source_path 
        and store an h5 file in config.data_sink_path/{self.config.dataset_name}.h5
        with this schema :
        {
            "train_vectors": train_data (train_ratio*len(train_data) , dim)
            "test_vectors": test_data (len(test_data) , dim)
            "distances_batch_{i}": test_distance (len(test_data) , len(train_data/i*batch))
        }
        """
        train_vectors, test_vectors, distance_type = self.get_raw_data()
        train_start, train_end = self.config.train_batch
        train_vectors = train_vectors[train_start:train_end]
        train_vectors = train_vectors.astype(np.float32)
        test_vectors = test_vectors.astype(np.float32)
        vect_distance = vectorized_cosine_distances if distance_type == "cosine" else vectorized_eucliden_distance
        print("distance", distance_type)
        print("train_vectors", train_vectors.shape)
        print("test_vectors", test_vectors.shape)
        print("ratio", self.config.train_batch)
        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'a') as f:
            try:
                f.create_dataset('train_vectors', data=train_vectors)
                f.create_dataset('test_vectors', data=test_vectors)
            except:
                del f["train_vectors"]
                del f["test_vectors"]
                f["train_vectors"] = train_vectors
                f["test_vectors"] = test_vectors

        # compute distance to train data
        batch_size = 50_000
        # Loop through chunks of vectors from train
        for i in tqdm(range(0, len(train_vectors), batch_size)):
            chunk = train_vectors[i:i+batch_size]
            # Compute distances for the current chunk
            distances = vect_distance(chunk, test_vectors).T
            # print(distances.shape,distances[:10])
            # Flush distances to disk
            dataset_name = f"distances_batch_{i}"
            np.save(f"{self.config.data_sink_path}/{dataset_name}.npy", distances)
            self.files_to_load.append(dataset_name)

    def create_attr_vectors(self, vector_count: int):
        """
        Create attribute vectors for label data using single-attribute.

        Args:
            vector_count (int): The number of attribute vectors to create.

        Returns:
            numpy.ndarray: An array of attribute vectors with shape (vector_count, total_attributes).
        """
        # label data using single - attribute

        attr_vectors = np.zeros(
            (vector_count, sum(
                [int(1/selectivity) for selectivity in self.config.selectivity_range])),
            dtype=np.uint8
        )
        start = 0
        for selectivity in self.config.selectivity_range:
            # partition train data ids
            attr_vals = list(range(int(1/selectivity)))
            attr_vals = [
                np.eye(len(attr_vals))[val]
                for val in attr_vals
            ]
            for i in range(0, vector_count):
                val = random.choice(attr_vals)
                for j in range(0, len(val)):
                    attr_vectors[i][j+start] = val[j]
            start += len(attr_vals)
        return attr_vectors

    def create_train_attr_vectors(self):
        train_len = h5py.File(
            f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'r')["train_vectors"].shape[0]

        train_attr_vectors = self.create_attr_vectors(train_len)
        # Open the existing HDF5 file in append mode
        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'a') as hf:
            # Add the new dataset to the HDF5 file
            try:
                hf.create_dataset('train_attr_vectors',
                                  data=train_attr_vectors)
            except:
                del hf["train_attr_vectors"]
                hf["train_attr_vectors"] = train_attr_vectors

    def create_test_attr_vectors(self):
        test_len = h5py.File(
            f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'r')["test_vectors"].shape[0]

        test_attr_vectors = self.create_attr_vectors(test_len)
        # Open the existing HDF5 file in append mode
        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'a') as hf:
            # Add the new dataset to the HDF5 file
            hf.create_dataset('test_attr_vectors', data=test_attr_vectors)

    def get_valid_neighbors(self):
        """
        This function retrieves the valid neighbors for a given dataset. 
        It reads the test and train attribute vectors from an HDF5 file 
        and calculates the valid train neighbors based on the dot product 
        between the test and train attribute ranges. The valid train neighbors 
        are then saved to a temporary numpy file.

        Returns:
            None
        """
        test_attr = h5py.File(
            f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'r')["test_attr_vectors"]
        train_attr = h5py.File(
            f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'r')["train_attr_vectors"]

        start = 0
        for selectivity in tqdm(self.config.selectivity_range, desc="collecting candidates per selectivity"):
            # print(selectivity)
            end = start + int(1/selectivity)
            test_attr_range = test_attr[:, start:end]
            train_attr_range = train_attr[:, start:end]
            valid_train = dot_product_batch(
                test_attr_range, train_attr_range
            )  # mask_shape (test, train)
            np.save(f"{self.config.data_sink_path}/temp_valid_neighbors_{selectivity}.npy",
                    valid_train)

            start = end

    def collect_neighbors(self, start, end):
        """
        Collects neighbors for a given range of indices.

        Parameters:
            start (int): The starting index of the range.
            end (int): The ending index of the range.

        Returns:
            None
        """

        partitions = [
            np.load(f"{self.config.data_sink_path}/{dataset_name}.npy",
                    mmap_mode='r')[start:end]
            for dataset_name in self.files_to_load
        ]
        test_distance = np.concatenate(partitions, axis=1)

        for selectivity in self.config.selectivity_range:
            # print(selectivity)
            # mask_shape (test, train)
            valid_train = np.load(
                f"{self.config.data_sink_path}/temp_valid_neighbors_{selectivity}.npy", mmap_mode='r')[start:end]
            neighbors = masked_sort(
                test_distance, valid_train, top_k=100)  # (test, topk)

            np.save(f"{self.config.data_sink_path}/temp_neighbors_{selectivity}_{start}.npy",
                    neighbors)

    def get_true_neighbors(self):
        """
        Generates the true neighbors for each test vector.

        This function collects the true neighbors for each test vector in the dataset.
        It first determines the shape of the test vectors by accessing the HDF5 file.
        Then, it calls the `get_valid_neighbors()` method to obtain the valid neighbors per selectivity.
        Next, it sorts the neighbors based on distance and takes the top_k.
        The function iterates through the test vectors in batches and calls the `collect_neighbors()` method to collect the neighbors for each batch.
        After collecting the neighbors, the function aggregates and cleans them for each selectivity.
        It removes the temporary files that were created during the aggregation process.
        Finally, it saves the aggregated neighbors for each selectivity in the HDF5 file.

        Parameters:
        - self: The instance of the class.

        Returns:
        None
        """

        test_batch_size = 1000
        test_vecs_shape = h5py.File(
            f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'r')["test_vectors"].shape[0]

        #  valid neighbors per selectivity
        self.get_valid_neighbors()
        #  sort neighbors per selectivity using diastance and take top_k
        for i in tqdm(range(0, test_vecs_shape, test_batch_size), desc="collecting neighbors"):
            # print(i)
            self.collect_neighbors(start=i, end=i + test_batch_size)

        # agg and clean
        for selectivity in self.config.selectivity_range:
            file_path = f"{self.config.data_sink_path}/temp_valid_neighbors_{selectivity}.npy"
            os.remove(file_path)
            all_neighbors = []

            for i in tqdm(range(0, test_vecs_shape, test_batch_size), desc="aggregating neighbors ..."):
                neighbors = np.load(
                    f"{self.config.data_sink_path}/temp_neighbors_{selectivity}_{i}.npy")
                all_neighbors.append(neighbors)

                # Remove the individual file after loading its content
                file_path = f"{self.config.data_sink_path}/temp_neighbors_{selectivity}_{i}.npy"
                os.remove(file_path)

            # Concatenate results for the current selectivity
            aggregated_neighbors = np.concatenate(all_neighbors, axis=0)

            #  add batch_start_ref
            aggregated_neighbors = aggregated_neighbors + \
                self.config.train_batch[0]

            # Save the aggregated neighbors for the current selectivity
            with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'a') as hf:
                # Add the new dataset to the HDF5 file
                nei_selec_name = f"neighbors_selectivity_{str(selectivity).replace('.','_')}"
                hf.create_dataset(
                    f"{nei_selec_name}_{self.config.train_batch}", data=aggregated_neighbors)

                if nei_selec_name in hf:
                    old_neighbors = np.array(hf[nei_selec_name])
                    del hf[nei_selec_name]
                    hf[nei_selec_name] = np.concatenate(
                        [old_neighbors, aggregated_neighbors], axis=1)
                else:
                    hf.create_dataset(nei_selec_name,
                                      data=aggregated_neighbors)

    def clean(self):
        """
        Delete the "distances_batch" dataset from sink flder.
        """
        for dataset in self.files_to_load:
            os.remove(f"{self.config.data_sink_path}/{dataset}.npy")
        # move train and train_attr
        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'a') as hf:
            if "train_vectors_full" in hf:
                old_train_vectors = np.array(hf["train_vectors_full"])
                old_train_attr_vectors = np.array(
                    hf["train_attr_vectors_full"])
                del hf["train_vectors_full"]
                del hf["train_attr_vectors_full"]
                hf["train_vectors_full"] = np.concatenate(
                    [old_train_vectors, hf["train_vectors"]], axis=0)
                hf["train_attr_vectors_full"] = np.concatenate(
                    [old_train_attr_vectors, hf["train_attr_vectors"]], axis=0)
            else:
                hf.create_dataset("train_vectors_full",
                                  data=hf["train_vectors"])
                hf.create_dataset("train_attr_vectors_full",
                                  data=hf["train_attr_vectors"])

        pass

    def create(self):
        """
        Creates a new dataset by performing the following steps:

        1. Checks if the dataset file already exists in the specified data sink path. If it does, raises a FileExistsError.
        2. Normalizes the data by calling the `normalize_data` method.
        3. Creates train attribute vectors by calling the `create_train_attr_vectors` method.
        4. Creates test attribute vectors by calling the `create_test_attr_vectors` method.
        5. Obtains true neighbors by calling the `get_true_neighbors` method.
        6. Cleans up the dataset by calling the `clean` method.

        This function does not have any parameters.

        Raises:
            FileExistsError: If the dataset file already exists in the data sink path.

        Returns:
            None
        """

        # if "{self.config.dataset_name}.h5" in os.listdir(self.config.data_sink_path):
        #     raise FileExistsError(
        #         f"{self.config.data_sink_path}/{self.config.dataset_name}.h5 already exists. Please delete it and try again."
        #     )

        logger.info("Starting data normalization process...")
        self.normalize_data()
        logger.info("Data normalization completed.")

        logger.info("Creating train attribute vectors...")
        self.create_train_attr_vectors()
        logger.info("Train attribute vectors created.")
        if 0 in self.config.train_batch:
            logger.info("Creating test attribute vectors...")
            self.create_test_attr_vectors()
            logger.info("Test attribute vectors created.")

        logger.info("Getting true neighbors...")
        self.get_true_neighbors()
        logger.info("True neighbors obtained.")

        logger.info("Cleaning up...")
        self.clean()
        logger.info("Done creating normalized data")
