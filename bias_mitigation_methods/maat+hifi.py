"""
The distinguished debiasing method MAAT is combined with our in-processing method HIFI to further boost effectiveness.
In this implementation, we assume that MAAT models and HIFI-regularized models are already trained.
"""


import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import copy
import math
import json
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib

from aif360.datasets import BinaryLabelDataset

from tools.utils import set_seed, makedirs
from tools.config import SPLIT_RATIO, considered_sensitive_attributes
from tools.get_classifier import get_classifier
from tools.metrics import measure_final_score

from models.model_utils import load_original_model, model_standardize


def train_with_maat_hifi(dataset_name, classifier_name, seed_range=[0], eta=0.75):
    method_name = "maat+hifi"
    sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
    split_ratio = SPLIT_RATIO[dataset_name]

    scaler = MinMaxScaler()

    dataset_orig = pd.read_csv(
        osp.join(osp.dirname(__file__), "../data/tabular", dataset_name, dataset_name + "_processed.csv"))

    for seed in seed_range:
        set_seed(seed)
        save_root = osp.join(osp.dirname(__file__), "../models", dataset_name, method_name, "seed_" + str(seed), f"eta={eta}")
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

        # Load trained MAAT models
        MAAT_orig_model = load_original_model("maat", dataset_name, classifier_name, seed)
        MAAT_standard_model, _ = model_standardize("maat", dataset_name, classifier_name, MAAT_orig_model)
        MAAT_pred_probs = MAAT_standard_model(dataset_orig_test.features)

        # Load trained HIFI-regularized model
        HIFI_name = "hifi_eta=" + str(eta)
        HIFI_orig_model = load_original_model(HIFI_name, dataset_name, classifier_name, seed)
        HIFI_standard_model, _ = model_standardize(HIFI_name, dataset_name, classifier_name, HIFI_orig_model)
        HIFI_pred_probs = HIFI_standard_model(dataset_orig_test.features)

        prob_avg = (MAAT_pred_probs * (len(sensitive_attributes) + 1) + HIFI_pred_probs) / (len(sensitive_attributes) + 2)
        ensemble_labels = np.where(prob_avg >= 0.5, 1, 0)

        test_df_copy = copy.deepcopy(dataset_orig_test)
        test_df_copy.labels = ensemble_labels

        round_result = measure_final_score(dataset_orig_test, test_df_copy, sensitive_attributes)

        with open(osp.join(save_root, classifier_name + "_metrics.json"), "w") as f:
            json.dump(round_result, f, indent=1)


if __name__ == '__main__':
    datasets = ["census", "ufrgs", "compas", "diabetes", "default"]
    classifiers = ["lr", "svm", "dl"]
    seed_range = list(range(0, 10))
    list_eta = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100, 1000]

    n_process = multiprocessing.cpu_count()  # number of parallel processes
    with multiprocessing.Pool(processes=n_process) as pool:
        for dataset in datasets:
            for clf in classifiers:
                for seed in seed_range:
                    for eta in list_eta:
                        pool.apply_async(train_with_maat_hifi, args=(dataset, clf, [seed], eta))

        pool.close()
        pool.join()