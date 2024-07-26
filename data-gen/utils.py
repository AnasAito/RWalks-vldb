import math
from dataclasses import dataclass
import numpy as np
import random
import h5py
import os
from tqdm import tqdm
import numba
from loguru import logger
import sys
import hnswlib
import time


log_file_path = 'logs.log'  # Change this to your desired log file path

# Configure logger to write to both console and file
logger.remove()  # Remove any previous handlers (optional)
logger.add(sys.stdout, colorize=True,
           format="<green>{time}</green> <level>{message}</level>")
logger.add(log_file_path, rotation="10 MB", compression="zip",
           format="{time:YYYY-MM-DD HH:mm:ss} {level} - {message}")


@dataclass
class FANNConfig:
    data_source_path: str
    data_sink_path: str
    dataset_name: str
    distance_type: str
    train_ratio: float
    selectivity_range: list


class ZipfDistribution:
    def __init__(self, num_points, num_labels):
        self.num_labels = num_labels
        self.num_points = num_points
        self.distribution_factor = 0.7

    def create_distribution_map(self):
        distribution_map = {}
        primary_label_freq = math.ceil(
            self.num_points * self.distribution_factor)
        for i in range(1, self.num_labels + 1):
            distribution_map[i] = math.ceil(primary_label_freq / i)
        return distribution_map

    def write_distribution(self, outfile):
        distribution_map = self.create_distribution_map()
        print("label distribution = ", distribution_map)
        for i in range(self.num_points):
            label_written = False
            for label, freq in distribution_map.items():
                label_selection_probability = random.random() < self.distribution_factor / label
                if label_selection_probability and freq > 0:
                    if label_written:
                        outfile.write(',')
                    outfile.write(str(label))
                    label_written = True
                    distribution_map[label] -= 1
            if not label_written:
                outfile.write('0')
            if i < self.num_points - 1:
                outfile.write('\n')

    def parse_labels_file(self, filename):
        labels = []
        with open(filename, 'r') as infile:
            for line in infile:
                row = [int(label) for label in line.strip().split(',')]
                labels.append(row)
        return labels

    def one_hot_encode(self, labels, num_unique_labels):
        num_points = len(labels)
        encoded_labels = np.zeros((num_points, num_unique_labels), dtype=int)
        for i in range(num_points):
            #  universal label
            if labels[i] == [0]:
                #  all entries are 1
                # print("universal label", )
                encoded_labels[i, :] = 1
                # print(labels[i], encoded_labels[i, :])
                # print(labels[i-4], encoded_labels[i-4, :])
            for label in labels[i]:
                if label != 0:
                    encoded_labels[i, label - 1] = 1
        return encoded_labels


class FannDataLoader:
    def __init__(self, config: FANNConfig):
        self.config = config
        self.files_to_load = []
        self.train_count = None
        self.test_count = None
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
        train_vectors = train_vectors[:int(
            len(train_vectors)*self.config.train_ratio)]
        train_vectors = train_vectors.astype(np.float32)
        test_vectors = test_vectors.astype(np.float32)
        # vect_distance = vectorized_cosine_distances if distance_type == "cosine" else vectorized_eucliden_distance
        print("distance", distance_type)
        print("train_vectors", train_vectors.shape)
        print("test_vectors", test_vectors.shape)
        print("ratio", self.config.train_ratio)
        self.train_count = len(train_vectors)
        self.test_count = len(test_vectors)
        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'w') as f:
            f.create_dataset('train_vectors', data=train_vectors)
            f.create_dataset('test_vectors', data=test_vectors)

    def get_selectivity(self, val):
        return 0.4 * np.exp(-1.9*val)

    def create_attr_vectors(self, vector_count: int, attr_count: int, is_test=False):
        # bit_width = 1.5 / attr_count
        # vals = [bit_width * i for i in range(attr_count)]

        # selectivities = [round(self.get_selectivity(val), 3) for val in vals]
        # print("selectivities = ", selectivities)
        # attr_vectors = np.zeros(
        #     (vector_count, attr_count), dtype=np.int32
        # )
        # if not is_test:
        #     for idx, s in enumerate(selectivities):
        #         attr_vectors[:, idx:idx +
        #                      1] = np.random.binomial(1, s, (vector_count, 1))
        # else:
        #     workload_ratio = int(vector_count / attr_count)
        #     for idx, s in enumerate(selectivities):
        #         attr_vectors[idx*workload_ratio:(idx + 1)*workload_ratio, idx:idx +
        #                      1] = np.ones((workload_ratio, 1))

        if is_test:
            attr_vectors = np.zeros(
                (vector_count, attr_count), dtype=np.int32
            )
            workload_ratio = int(vector_count / attr_count)
            for idx, s in enumerate(range(attr_count)):
                attr_vectors[idx*workload_ratio:(idx + 1)*workload_ratio, idx:idx +
                             1] = np.ones((workload_ratio, 1))
            return attr_vectors
        else:
            output_file = f"{self.config.data_sink_path}/{self.config.dataset_name}_labels.txt"

            with open(output_file, 'w') as outfile:
                # distribution_type == "zipf":
                zipf = ZipfDistribution(vector_count, attr_count)
                zipf.write_distribution(outfile)

            print(f"Labels written to {output_file}")
            # Read labels from file and transform
            parsed_labels = zipf.parse_labels_file(output_file)
            encoded_labels = zipf.one_hot_encode(parsed_labels, attr_count)
            print("Parsed and one-hot encoded labels shape:", encoded_labels.shape)
            print("Example one-hot encoded label:", encoded_labels[0])

            return encoded_labels

    def get_neighbors(self):
        # load data
        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'r') as f:
            train = np.array(f['train_vectors'])  # [:100]
            test = np.array(f['test_vectors'])  # [:2]
            train_attr = np.array(f['train_attr_vectors'])  # [:100]
            test_attr = np.array(f['test_attr_vectors'])  # [:2]

        #  init index
        index = hnswlib.BFIndex(space="l2",
                                dim=train.shape[1],
                                dim_attr=train_attr.shape[1],
                                )
        index.init_index(
            max_elements=train.shape[0])
        #  build index
        print("Start Build Index")
        start = time.time()
        print("shapes", train.shape, train_attr.shape)
        index.add_items(train, train_attr)
        print("Finish Build Index", time.time()-start, "s")
        print("Start Query")
        start = time.time()
        labels, distances = index.knn_query(
            test, test_attr, k=100, num_threads=4)

        # print(labels[:100])
        time_s = time.time() - start
        print("Finish Query", time_s, "s")
        print("QPS = ", test.shape[0] / time_s)
        print(labels.shape, labels.dtype)
        return labels, distances

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

        if "{self.config.dataset_name}.h5" in os.listdir(self.config.data_sink_path):
            raise FileExistsError(
                f"{self.config.data_sink_path}/{self.config.dataset_name}.h5 already exists. Please delete it and try again."
            )

        logger.info("Starting data normalization process...")
        self.normalize_data()
        logger.info("Data normalization completed.")

        logger.info("Creating train attribute vectors...")
        attr_vecs_train = self.create_attr_vectors(self.train_count, 100)
        logger.info("Train attribute vectors created.")

        logger.info("Creating test attribute vectors...")
        attr_vecs_test = self.create_attr_vectors(
            self.test_count, 100, is_test=True)
        logger.info("Test attribute vectors created.")

        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'a') as f:
            f.create_dataset('train_attr_vectors', data=attr_vecs_train)
            f.create_dataset('test_attr_vectors', data=attr_vecs_test)

        logger.info("Getting true neighbors...")
        labels, distances = self.get_neighbors()
        logger.info("True neighbors obtained.")

        with h5py.File(f"{self.config.data_sink_path}/{self.config.dataset_name}.h5", 'a') as f:
            f.create_dataset('neighbors', data=labels)
            f.create_dataset('distances', data=distances)

        # logger.info("Cleaning up...")
        # self.clean()
        # logger.info("Done creating normalized data")
