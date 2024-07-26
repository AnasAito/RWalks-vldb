# RWalks

Filtered Approximate nearest neighbor search using graph walks as attribute diffusers

## Create an datset for Filtered - ANN

- Arxiv dataset (2M vector, 384 dim, cosine distance)

First download the raw dataset from [qdrant/ann-filtering-benchmark-datasets](https://github.com/qdrant/ann-filtering-benchmark-datasets), Then use this code snippet :

```python
config = FANNConfig(
    data_source_path="/Users/mac/Downloads/arxiv",
    data_sink_path="/Users/mac/dev/gwad-ann/data/arxiv",
    dataset_name="arxiv_100k",
    distance_type="cosine",
    train_ratio=0.05,
    selectivity_range=[0.05, 0.1, 0.2, 0.3, 0.4],
)

arxiv_data_loader = ArxivDataLoader(config)
arxiv_data_loader.create()

data = h5py.File(
    f"{config.data_sink_path}/{config.dataset_name}.h5", 'r')
for key in data.keys():
    print(key)
    print(data[key].shape)
```

- Sift1M (1M vector, 128 dim, eucliden distance)

```python
config = FANNConfig(
    data_source_path="/Users/mac/Downloads/sift_base.tar.gz",
    data_sink_path="/Users/mac/dev/gwad-ann/data/sift",
    dataset_name="sift_50k",
    distance_type="l2",
    train_ratio=0.05,
    selectivity_range=[0.05, 0.1, 0.2, 0.3, 0.4],
)

arxiv_data_loader = Sift1MDataLoader(config)
arxiv_data_loader.create()

data = h5py.File(
    f"{config.data_sink_path}/{config.dataset_name}.h5", 'r')
for key in data.keys():
    print(key)
    print(data[key].shape)

```

## Run experiment :

```shell
$python3 run.py \
        --index_name hnsw-base \
        --data_dir /Users/mac/dev/gwad-ann/data/sift/sift_200k.h5 \
        --dts_type cosine \
        --selectivity 0.2 \
        --use_attr_in_train True \
        --M 5 \
        --ef_construction 100

loading data to memory (/Users/mac/dev/gwad-ann/data/sift/sift_200k.h5) ...
2023-12-06 21:11:54.932 | INFO     | context:load_data:104 - 📊 Data loaded successfully. Samples: 200000, Dimensionality: 128
2023-12-06 21:12:09.795 | INFO     | context:build_index:125 - 🏗️ Index built successfully in 14.86 seconds -  Index Size: 0.25 GB
2023-12-06 21:12:09.796 | INFO     | context:get_stats:129 - 🧪 Evaluating index ...
2023-12-06 21:12:09.979 | INFO     | context:get_stats:154 - 🧪 Evaluation completed.
Stats:--------------------
{'qps': 81026, 'recalls': {'top10': 0.05}}
---------------------------
2023-12-06 21:12:10.003 | INFO     | context:get_deep_stats:161 - 🧪 Deep Evaluation started ...
2023-12-06 21:12:10.522 | INFO     | context:get_deep_stats:182 - 🧪 Deep Evaluation completed.
Deep stats (Latency go brrrr) :--------------------
{
    'qps': 21777,
    'recalls': {'top10': 0.05},
    'valid_ratio': {
        'valid_ratio_max': 0.0,
        'valid_ratio_min': 0.0,
        'valid_ratio_mean': 0.0,
        'valid_ratio': array([0., 0., 0., ..., 0., 0., 0.], dtype=float32)
    },
    'nhops': {
        'nhops_max': 31,
        'nhops_min': 10,
        'nhops_mean': 13.3877,
        'nhops': array([12, 15, 16, ..., 11, 13, 11], dtype=int32)
    }
}
---------------------------
```
