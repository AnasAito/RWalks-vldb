import argparse
from dataclasses import dataclass
from context import IndexPerformanceEvaluator
from rich import print as rprint
from typing import List
import h5py
import uuid
from itertools import product

"""
Example of index building, search and serialization/deserialization
"""


@dataclass
class ExperimentConfig:
    index_name: str
    data_dir: str
    dst_type: str
    selectivity: float
    use_attr_in_train: bool
    index_params: dict
    search_params: List[dict]


def store_object_to_hdf5(obj, filename):
    with h5py.File(f"/Users/mac/dev/GWAD/exps_logs/{filename}.h5", 'a') as f:
        if not f:
            f.create_group('experiments')

        exp_hash = uuid.uuid4().hex
        group = f.create_group(
            f'experiments/exp_{obj["M"]}_{obj["ef_construction"]}_{obj["ef"]}_{exp_hash}')
        group.create_dataset('qps_4_threads', data=obj['qps_4_threads'])
        group.create_dataset('recalls/top10', data=obj['recalls']['top10'])
        group.create_dataset('recalls/top10_data',
                             data=obj['recalls']['top10_data'])
        group.create_dataset('qps_1_threads', data=obj['qps_1_threads'])

        valid_ratio = obj['valid_ratio']
        valid_ratio_group = group.create_group('valid_ratio')
        valid_ratio_group.create_dataset(
            'valid_ratio_max', data=valid_ratio['valid_ratio_max'])
        valid_ratio_group.create_dataset(
            'valid_ratio_min', data=valid_ratio['valid_ratio_min'])
        valid_ratio_group.create_dataset(
            'valid_ratio_mean', data=valid_ratio['valid_ratio_mean'])
        valid_ratio_group.create_dataset(
            'valid_ratio_data', data=valid_ratio['valid_ratio'])

        nhops = obj['nhops']
        nhops_group = group.create_group('nhops')
        nhops_group.create_dataset('nhops_max', data=nhops['nhops_max'])
        nhops_group.create_dataset('nhops_min', data=nhops['nhops_min'])
        nhops_group.create_dataset('nhops_mean', data=nhops['nhops_mean'])
        nhops_group.create_dataset('nhops_data', data=nhops['nhops'])

        group.create_dataset('k', data=obj['k'])
        group.create_dataset('ef', data=obj['ef'])
        group.create_dataset('hybrid_factor', data=obj['hybrid_factor'])
        group.create_dataset('pron_factor', data=obj['pron_factor'])
        group.create_dataset('M', data=obj['M'])
        group.create_dataset('ef_construction', data=obj['ef_construction'])

        group.create_dataset('selectivity', data=obj['selectivity'])
        group.create_dataset('index_size', data=obj['index_size'])
        group.create_dataset('built_time', data=obj['built_time'])

        # nhops = obj['neighbors']
        # nhops_group = group.create_group('neighbors')
        # nhops_group.create_dataset('neighbors', data=nhops['neighbors'])
        # nhops_group.create_dataset(
        #     'neighbors_mapped', data=nhops['neighbors_mapped'])
        # nhops_group.create_dataset('selectivity', data=nhops['selectivity'])
        # nhops_group.create_dataset('distances', data=nhops['distances'])


def run_experiment(exp_config):
    evaluator = IndexPerformanceEvaluator(
        index_name=exp_config.index_name,
        data_dir=exp_config.data_dir,
        dst_type=exp_config.dst_type,
        use_attr_in_train=exp_config.use_attr_in_train,
        selectivity=exp_config.selectivity
    )
    results = []
    evaluator.load_data()
    evaluator.build_index(exp_config.index_params)
    for search_params_atom in exp_config.search_params:
        stats = evaluator.get_stats(search_params_atom)
        print("Stats:--------------------")
        rprint(stats)
        print("---------------------------")
        # deep_stats = evaluator.get_deep_stats(search_params_atom)
        # print("Deep stats (Latency go brrrr) :--------------------")
        # rprint(deep_stats["valid_ratio"])
        # rprint(deep_stats["nhops"])
        # print("---------------------------")
        # results.append({**stats, **deep_stats, **search_params_atom,
        #                **exp_config.index_params,
        #                **{"selectivity": exp_config.selectivity,
        #                 "index_size": evaluator.data_stats["index_size"],
        #                   "built_time": evaluator.data_stats["built_time"]}})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run experiment with specified configuration')
    parser.add_argument('--index_name', type=str,
                        default='hnsw-base', help='Name of the index')
    parser.add_argument('--h5_sink_path', type=str,
                        default='dummy_exp.h5',
                        help='Path to store results (H5 file)')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/mac/dev/RWalks-vldb/data/sift_50k.h5', help='Directory containing data')
    parser.add_argument('--dst_type', type=str,
                        default='l2', help='distance type')
    parser.add_argument('--selectivity', type=float,
                        default=0.05, help='Selectivity value')
    parser.add_argument('--use_attr_in_train', type=bool,
                        default=False, help='Whether to use attribute in training')
    parser.add_argument('--M', type=int, default=32, help='Value for M')
    parser.add_argument('--ef_construction', type=int,
                        default=200, help='Value for ef_construction')
    parser.add_argument('--efs', type=str,
                        default="10,50,100,200,400,500", help='Values for ef (separeted by comma)')
    parser.add_argument('--hybrid_factors', type=str,
                        default="default", help='Values for dist regulation factor (separeted by comma)')
    parser.add_argument('--pron_factors', type=str,
                        default="default", help='Values for pronning factor (separeted by comma)')
    args = parser.parse_args()

    index_params = {"M": args.M,
                    "ef_construction": args.ef_construction}

    search_params = list(product(
        [int(ef)
         for ef in args.efs.strip().split(",")],
        [float(hybrid) for hybrid in args.hybrid_factors.strip().split(
            ",")] if args.hybrid_factors != "default" else [None],
        [float(pron) for pron in args.pron_factors.strip().split(",")] if args.pron_factors != "default" else [None]))
    search_params = [dict(zip(["ef", "hybrid_factor", "pron_factor"], param))
                     for param in search_params]
    search_params = [dict(**{"k": 10}, **param) for param in search_params]
    # search_params = [
    #     parm_comb for parm_comb in search_params
    #     if not (parm_comb["hybrid_factor"] == 0 and parm_comb["pron_factor"] != 0)
    # ]
    print("Search params: ", len(search_params))
    rprint(search_params)
    exp_config = ExperimentConfig(
        index_name=args.index_name,
        data_dir=args.data_dir,
        selectivity=args.selectivity,
        dst_type=args.dst_type,
        use_attr_in_train=args.use_attr_in_train,
        index_params=index_params,
        search_params=search_params

    )
    results = run_experiment(exp_config)
    for exp_result in results:
        store_object_to_hdf5(exp_result, args.h5_sink_path)
