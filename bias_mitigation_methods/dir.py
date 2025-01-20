"""
This code is partially adapted from:
- https://github.com/Trusted-AI/AIF360/blob/main/examples/demo_disparate_impact_remover.ipynb
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Fair3602/dir.py

Disparate-Impact Remover (DIR) increases group fairness by ensuring balanced representation of different protected groups.
The DIR algorithm works by adjusting the features in the dataset so the distribution of these features is equal across all protected groups.
"""


import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import copy
import json
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover

from tools.utils import set_seed, makedirs
from tools.config import SPLIT_RATIO, considered_sensitive_attributes
from tools.get_classifier import get_classifier
from tools.metrics import measure_final_score


def train_with_dir(dataset_name, classifier_name, seed_range=[0]):
    method_name = "dir"
    sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
    indices_sensitive_attributes = list(considered_sensitive_attributes[dataset_name].values())
    split_ratio = SPLIT_RATIO[dataset_name]

    scaler = MinMaxScaler()

    dataset_orig = pd.read_csv(
        osp.join(osp.dirname(__file__), "../data/tabular", dataset_name, dataset_name + "_processed.csv"))

    for seed in seed_range:
        set_seed(seed)
        save_root = osp.join(osp.dirname(__file__), "../models", dataset_name, method_name, "seed_" + str(seed))
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

        di = DisparateImpactRemover(sensitive_attribute=sensitive_attributes)
        train_repd = di.fit_transform(dataset_orig_train)
        test_repd = di.fit_transform(dataset_orig_test)
        X_train_reduced = np.delete(train_repd.features, indices_sensitive_attributes, axis=1)
        X_test_reduced = np.delete(test_repd.features, indices_sensitive_attributes, axis=1)
        y_train = train_repd.labels.ravel()

        clf = get_classifier(classifier_name, X_train_reduced.shape[1:])
        if classifier_name == "dl":
            clf.fit(X_train_reduced, y_train, epochs=20)
            preds = (clf.predict(X_test_reduced) > 0.5).astype("int32")
            clf.save(osp.join(save_root, classifier_name+".keras"))
        else:
            clf = clf.fit(X_train_reduced, y_train)
            preds = clf.predict(X_test_reduced).reshape(-1, 1)
            joblib.dump(clf, osp.join(save_root, classifier_name+".pkl"))

        test_df_copy = copy.deepcopy(test_repd)
        test_df_copy.labels = preds

        round_result = measure_final_score(test_repd, test_df_copy, sensitive_attributes)

        with open(osp.join(save_root, classifier_name+"_metrics.json"), "w") as f:
            json.dump(round_result, f, indent=1)


if __name__ == '__main__':
    datasets = ["census", "ufrgs", "compas", "diabetes", "default"]
    classifiers = ["lr", "svm", "rf", "dl"]
    seed_range = list(range(0, 10))

    n_process = multiprocessing.cpu_count()  # number of parallel processes
    with multiprocessing.Pool(processes=n_process) as pool:
        for dataset in datasets:
            for clf in classifiers:
                for seed in seed_range:
                    pool.apply_async(train_with_dir, args=(dataset, clf, [seed]))

        pool.close()
        pool.join()