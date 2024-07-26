import tarfile
import numpy
from typing import Any, Callable, Dict, Tuple
from utils import FannDataLoader, FANNConfig

import h5py
# "/Users/mac/Downloads/arxiv/vectors.npy"


def _load_texmex_vectors(f: Any, n: int, k: int) -> numpy.ndarray:
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str) -> numpy.ndarray:
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def get_sift(fn: str) -> None:
    import tarfile
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
    return train, test


class Sift1MDataLoader(FannDataLoader):
    def __init__(self, config: FANNConfig):
        super().__init__(config)

    def get_raw_data(self):
        """
        Returns the raw train and test data along with the similarity metric used.

        Parameters:
            None

        Returns:
            train_vectors (ndarray): An array containing the training vectors.
            test_vectors (ndarray): An array containing the test vectors.
            distance_type (str): The distance used.
        """

        # prepare train data
        train, test = get_sift("/Users/mac/Downloads/sift_base.tar.gz")
        return train, test, "l2"


#  test
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import numba

    config = FANNConfig(
        data_source_path="/Users/mac/Downloads/sift_base.tar.gz",
        data_sink_path="/Users/mac/dev/GWAD/data",
        dataset_name="sift_1M_zipf_12",
        distance_type="l2",
        train_ratio=1,
        selectivity_range=[0.05, 0.1, 0.2, 0.3, 0.4],
    )

    sift_data_loader = Sift1MDataLoader(config)
    sift_data_loader.create()

    data = h5py.File(
        f"{config.data_sink_path}/{config.dataset_name}.h5", 'r')

    # check selectivities
    print("inspection--------------------")
    print("check selectivities")
    train_attr = data["train_attr_vectors"]
    train_attr_s = np.sum(train_attr, axis=0) / train_attr.shape[0]
    print(train_attr_s)
    print("check query workload size")
    test_attr = data["test_attr_vectors"]
    test_attr_s = np.sum(test_attr, axis=0) / test_attr.shape[0]
    print(test_attr_s)
