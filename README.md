# RWalks: Random Walks as Attribute Diffusers for Filtered Vector Search
The repo provides the code for RWalks and supplementary materials. 
## Installation

Follow these steps to set up the project:

### Step 1: Create and Activate a Virtual Environment

First, create a virtual environment and activate it.

#### On Windows:
```sh
python -m venv myenv
myenv\Scripts\activate
```
####  On macOS and Linux:
```sh
python3 -m venv myenv
source myenv/bin/activate
```
### Step 2: Install Requirements
With the virtual environment activated, install the required packages listed in requirements.txt.
```sh
pip install -r requirements.txt
```

### Step 3: Build and Install RWalks
Navigate to the indices/hnsw-base directory and run the following commands to clean and install the project.

```sh
cd indices/hnsw-base
make clean && pip install .
```

## Run experiment :
We provide a 50k subset of the Sift dataset to try the index building and search. you can provide your own data. We vary search parameter ef (10,100,500) and track QPS, and Recall.

```shell
$python3 run.py \
        --data_dir data/sift_50k_s_005.h5 \
        --dts_type l2 \
        --M 32 \
        --ef_construction 200
        --efs 10,100,500
        --hybrid_factors 0.06
        --pron_factors 0.0
```
```shell
loading data to memory (/Users/mac/dev/RWalks-vldb/data/sift_50k_s_005.h5) ...
2024-07-26 23:49:03.439 | INFO     | context:load_data:106 - 📊 Data loaded successfully. Samples: 50000, Dimensionality: 128
2024-07-26 23:49:57.467 | INFO     | context:build_index:159 - 🏗️ Index built successfully in 5.59 seconds -  Index Size: 0.05 GB
Search Params : {'k': 10, 'ef': 10, 'hybrid_factor': 0.06, 'pron_factor': 0.0}
2024-07-26 23:49:57.729 | INFO     | context:get_stats:198 - 🧪 Evaluation completed.
Stats:--------------------
{'qps_4_threads': 82223, 'recalls': {'top10': np.float64(0.243)}}
---------------------------
2024-07-26 23:49:57.730 | INFO     | context:get_deep_stats:209 - 🧪 Deep Evaluation started ...
2024-07-26 23:49:58.185 | INFO     | context:get_deep_stats:234 - 🧪 Deep Evaluation completed.
Deep stats (Latency go brrrr) :--------------------
{'valid_ratio_max': np.float32(1.0), 'valid_ratio_min': np.float32(1.0), 'valid_ratio_mean': np.float32(1.0), 'valid_ratio': array([1., 1., 1., ..., 1., 1., 1.], dtype=float32)}
{'nhops_max': np.int32(28), 'nhops_min': np.int32(1), 'nhops_mean': np.float64(12.6753), 'nhops': array([13, 11, 10, ..., 12, 10, 10], dtype=int32)}
---------------------------
2024-07-26 23:49:58.188 | INFO     | context:get_stats:174 - 🧪 Evaluating index ...
Search Params : {'k': 10, 'ef': 10, 'hybrid_factor': 0.06, 'pron_factor': 0.0}
2024-07-26 23:54:22.563 | INFO     | context:get_stats:198 - 🧪 Evaluation completed.
Stats:--------------------
{'qps_4_threads': 100565, 'recalls': {'top10': np.float64(0.242)}}
---------------------------
2024-07-26 23:54:22.564 | INFO     | context:get_stats:174 - 🧪 Evaluating index ...
Search Params : {'k': 10, 'ef': 100, 'hybrid_factor': 0.06, 'pron_factor': 0.0}
2024-07-26 23:54:23.105 | INFO     | context:get_stats:198 - 🧪 Evaluation completed.
Stats:--------------------
{'qps_4_threads': 20844, 'recalls': {'top10': np.float64(0.937)}}
---------------------------
2024-07-26 23:54:23.106 | INFO     | context:get_stats:174 - 🧪 Evaluating index ...
Search Params : {'k': 10, 'ef': 500, 'hybrid_factor': 0.06, 'pron_factor': 0.0}
2024-07-26 23:54:25.269 | INFO     | context:get_stats:198 - 🧪 Evaluation completed.
Stats:--------------------
{'qps_4_threads': 4752, 'recalls': {'top10': np.float64(0.997)}}
---------------------------

```

## Data Layout

For test purposes, data is provided as an h5 file containing 5 datasets: `neighbors`, `test_attr_vectors`, `test_vectors`, `train_attr_vectors`, and `train_vectors`.

- `train_vectors` and `test_vectors` are NumPy arrays of shape (N*d), where N is the number of vectors and d is the vector dimension.
- `train_attr_vectors` is a 2D NumPy array containing 0/1 values encoding the attributes. Its shape is (N*p), where p is the number of all possible attributes.
- `test_attr_vectors` same as train_attr_vectors containing query filters.
- `neighbors` is a 2D array containing the K-NN neighbors for each point.

