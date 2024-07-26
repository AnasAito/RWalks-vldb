import time

from loguru import logger
from tqdm import tqdm
from dataclasses import dataclass
import h5py

import matplotlib.pyplot as plt
import pprint
import json
import numpy as np
import psutil
from typing import List, Union
import hashlib
# index import
import hnswlib
import os
from rich import print as rprint


@dataclass
class DataSet:
    train: np.ndarray
    test: np.ndarray
    neighbors: np.ndarray
    train_attr: np.ndarray
    test_attr: np.ndarray


INDEX_STORE = {
    "hnsw-base": hnswlib.Index,

}


def get_memory_usage():
    """Returns the current memory usage of this ANN algorithm instance in kilobytes.

    Returns:
        float: The current memory usage in kilobytes (for backwards compatibility), or None if
            this information is not available.
    """

    return psutil.Process().memory_info().rss / 1024


class IndexPerformanceEvaluator:
    def __init__(self, index_name, data_dir, dst_type, selectivity, use_attr_in_train=False):

        self.index_store_dir = "/Users/mac/dev/gwad-ann/indices_store"
        # self.index_store_dir = "./indices_store"
        self.num_threads = 4
        self.index_name = index_name
        self.data_dir = data_dir
        self.selectivity = selectivity
        self.dst_type = dst_type
        self.use_attr_in_train = use_attr_in_train
        self.data = None
        self.built_index = None
        self.query_times = []
        self.data_stats = {
            "num_samples": 0,
            "dimensionality": 0,
        }

        logger.add("index_performance.log", rotation="10 MB",
                   format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}")
        logger.disable(None)
        self.progress_bar = None

    def h5_to_memory(self, exp_env_name, selectivity):


        with h5py.File(exp_env_name, 'r') as dataset:
            # Load datasets into memory
            print(dataset.keys())
            train = np.array(dataset["train_vectors"])
            train_attr = np.array(dataset["train_attr_vectors"])
            test = np.array(dataset["test_vectors"])
            test_attr = np.array(dataset["test_attr_vectors"])
            print(train.shape, train_attr.shape, test.shape, test_attr.shape)
            neighbors_key = f"neighbors"

            print(neighbors_key)
            neighbors = np.array(dataset[neighbors_key])
            print(neighbors.shape)
            return DataSet(
                train=train,  # [self.original_ids],
                test=test,
                train_attr=train_attr,  # [self.original_ids],
                test_attr=test_attr,
                neighbors=neighbors,
            )

    def load_data(self):
        if not self.data_dir:
            raise ValueError(
                "Data directory not provided. Please set data_dir.")
        print(f"loading data to memory ({self.data_dir}) ...")
        self.data = self.h5_to_memory(self.data_dir, self.selectivity)
        print(self.data.test_attr.shape)
        print(self.data.test_attr[0], self.data.test_attr[0].sum())
        self.data_stats["num_samples"] = self.data.train.shape[0]
        self.data_stats["dimensionality"] = self.data.train.shape[1]
        self.data_stats["dimensionality_attr"] = self.data.train_attr.shape[1]
        logger.info(
            f"📊 Data loaded successfully. Samples: {self.data_stats['num_samples']}, Dimensionality: {self.data_stats['dimensionality']}")
        return self

    def build_index(self, index_params, use_cache=False):
        index_signature = {
            "M": index_params["M"],
            "ef_construction": index_params["ef_construction"],
            "dataset_name": self.data_dir.split("/")[-1].split(".")[0],
            "index_build_type": "unconstrained",
            "walk": "20_10_1"
            # "walk": "norm_l2"
        }
        index_signature_hash = hashlib.md5(
            json.dumps(index_signature, sort_keys=True).encode()).hexdigest()
        index_path = f"{self.index_store_dir}/hnsw_{index_signature_hash}.bin"

        if f"hnsw_{index_signature_hash}.bin" in os.listdir(self.index_store_dir) and use_cache:
            logger.info(f"🔎 Loading index from {index_path} ...")
            self.built_index = INDEX_STORE[self.index_name](
                space=self.dst_type,
                dim=self.data_stats["dimensionality"],
                dim_attr=self.data_stats["dimensionality_attr"],
            )
            self.built_index.load_index(index_path)
            self.data_stats["built_time"] = -1
            self.data_stats["index_size"] = -1

        else:

            self.built_index = INDEX_STORE[self.index_name](
                space=self.dst_type,
                dim=self.data_stats["dimensionality"],
                dim_attr=self.data_stats["dimensionality_attr"],
            )
            self.built_index.init_index(
                max_elements=self.data_stats["num_samples"], **index_params)
            self.built_index.set_num_threads(self.num_threads)

            print("Index Params :", {
                "M": self.built_index.M,
                "efConstruction": self.built_index.ef_construction,
            })
            rprint("train_attr_stats",
                   self.data.train_attr[0], self.data.train_attr.shape, self.data.train_attr[0].sum())
            memory_usage_before = get_memory_usage()
            start_time = time.time()
            self.built_index.add_items(self.data.train, self.data.train_attr)
            build_time = time.time() - start_time
            index_size = get_memory_usage() - memory_usage_before
            self.data_stats["built_time"] = build_time
            self.data_stats["index_size"] = index_size
            index_size_to_render = index_size / 1024 / 1024
            logger.info(
                f"🏗️ Index built successfully in {build_time:.2f} seconds -  Index Size: {index_size_to_render:.2f} GB")
            # save the index

            logger.info(f"Saving index to {index_path} ...")
            # remove the index if it already exists
            if os.path.exists(index_path):
                os.remove(index_path)
                os.remove(index_path.replace(".bin", "_attr.bin"))
            self.built_index.save_index(index_path)

    def get_stats(self, search_params):
        k = search_params["k"]
        ef = search_params["ef"]

        logger.info("🧪 Evaluating index ...")
        if self.built_index is None:
            raise ValueError("You need to build the index first.")
        if "hybrid_factor" in search_params:
            self.built_index.set_hybrid_factor(search_params["hybrid_factor"])
        if "pron_factor" in search_params:
            self.built_index.set_pron_factor(search_params["pron_factor"])
        # Measure the query time and recall

        self.built_index.set_ef(ef)
        print("Search Params :", search_params)
        recalls = {'top10': [], 'top100': []}
        start_time = time.time()
        self.built_index.set_num_threads(self.num_threads)
        labels, distances = self.built_index.knn_query(
            self.data.test, self.data.test_attr, k=k)
        qps = int(self.data.test.shape[0] / (time.time() - start_time))
        for neighbors, true_neighbors in zip(labels, self.data.neighbors):
            recall_at_10 = len(np.intersect1d(
                true_neighbors[:k], neighbors)) / min(k, len(true_neighbors))

            recalls['top10'].append(recall_at_10)


        logger.info("🧪 Evaluation completed.")
        self.recalls_data = recalls['top10']
        return {

            f"qps_{self.num_threads}_threads": qps,
            "recalls": {"top10": round(np.mean(recalls['top10']), 3)},
        }

    def get_deep_stats(self, search_params):
        k = search_params["k"]
        ef = search_params["ef"]
        logger.info("🧪 Deep Evaluation started ...")
        if self.built_index is None:
            raise ValueError("You need to build the index first.")
        if "hybrid_factor" in search_params:
            self.built_index.set_hybrid_factor(search_params["hybrid_factor"])
        if "pron_factor" in search_params:
            self.built_index.set_pron_factor(search_params["pron_factor"])
        # Measure the query time and recall
        recalls = {'top10': [], 'top100': []}
        start_time = time.time()
        self.built_index.set_ef(ef)
        self.built_index.set_num_threads(1)
        labels, distances, nhops, valid_ratio, distances_count = self.built_index.knn_query(
            self.data.test, self.data.test_attr, k=k, collect_metrics=True)
        # print(distances_attr[:10], distances_attr.shape)
        qps = int(self.data.test.shape[0] / (time.time() - start_time))

        for neighbors, true_neighbors in zip(labels, self.data.neighbors):
            # neighbors_mapped = [
            #     self.id_mapping_org[i] for i in neighbors]
            recall_at_10 = len(np.intersect1d(
                true_neighbors[:k], neighbors)) / min(k, len(true_neighbors))

            recalls['top10'].append(recall_at_10)

        logger.info("🧪 Deep Evaluation completed.")

        return {
            "qps_1_threads": qps,
            "recalls": {"top10": round(np.mean(recalls['top10']), 3), "top10_data": self.recalls_data},
            "valid_ratio": {
                "valid_ratio_max": valid_ratio.max(),
                "valid_ratio_min": valid_ratio.min(),
                "valid_ratio_mean": valid_ratio.mean(),
                "valid_ratio": valid_ratio
            },
            "nhops": {
                "nhops_max": nhops.max(),
                "nhops_min": nhops.min(),
                "nhops_mean": nhops.mean(),
                "nhops": nhops
            },
            "distances_attr": {
                "distances_attr_max": distances_count.max(),
                "distances_attr_min": distances_count.min(),
                "distances_attr_mean": distances_count.mean(),
                "distances_attr": distances_count
            }
        }

