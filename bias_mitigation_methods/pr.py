"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Fair3602/default.py

PR (Prejudice Remover) is an in-processing technique that adds a discrimination-aware regularization term
to the learning objective.
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import json
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing.prejudice_remover import PrejudiceRemover

from tools.utils import set_seed, makedirs
from tools.config import SPLIT_RATIO, considered_sensitive_attributes
from tools.get_classifier import get_classifier
from tools.metrics import measure_final_score


def train_with_pr(dataset_name, seed_range=[0]):
    method_name = "pr"
    sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
    split_ratio = SPLIT_RATIO[dataset_name]

    scaler = MinMaxScaler()

    dataset_orig = pd.read_csv(osp.join(osp.dirname(__file__), "../data/tabular", dataset_name, dataset_name+"_processed.csv"))

    for seed in seed_range:
        set_seed(seed)
        save_root = osp.join(osp.dirname(__file__), "../models", dataset_name, method_name, "seed_"+str(seed))
        makedirs(save_root)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=split_ratio, shuffle=True)

        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                                label_names=['Probability'],
                                                protected_attribute_names=sensitive_attributes)
        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=sensitive_attributes)

        if len(sensitive_attributes) == 1:
            prejudice = PrejudiceRemover(sensitive_attr=sensitive_attributes[0], eta=25)
        else:
            prejudice = PrejudiceRemover(sensitive_attr=sensitive_attributes, eta=25)

        prejudice = prejudice.fit(dataset_orig_train)
        pred_prejudice = prejudice.predict(dataset_orig_test)
        joblib.dump(prejudice, osp.join(save_root, "lr.pkl"))

        round_result = measure_final_score(dataset_orig_test, pred_prejudice, sensitive_attributes)

        with open(osp.join(save_root, "lr_metrics.json"), "w") as f:
            json.dump(round_result, f, indent=1)


if __name__ == '__main__':
    datasets = ["census", "ufrgs", "compas", "diabetes", "default"]
    seed_range = list(range(0, 10))

    n_process = multiprocessing.cpu_count()  # number of parallel processes
    with multiprocessing.Pool(processes=n_process) as pool:
        for dataset in datasets:
            for seed in seed_range:
                pool.apply_async(train_with_pr, args=(dataset, [seed]))

        pool.close()
        pool.join()