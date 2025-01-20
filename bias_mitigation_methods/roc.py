"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Fair3602/ceo.py

ROC is a postprocessing technique that gives favorable outcomes to unprivileged groups and unfavorable outcomes
to privileged groups in a confidence band around the decision boundary with the highest uncertainty.
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import json
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

from tools.utils import set_seed, makedirs
from tools.config import SPLIT_RATIO, considered_sensitive_attributes, privileged_groups, unprivileged_groups
from tools.get_classifier import get_classifier
from tools.metrics import measure_final_score


def train_with_roc(dataset_name, classifier_name, seed_range=[0]):
    method_name = "roc"
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

        clf = get_classifier(classifier_name, dataset_orig_train.features.shape[1:])
        if classifier_name in ["lr", "rf", "svm"]:
            if classifier_name == 'svm':
                clf = CalibratedClassifierCV(estimator=clf)
            clf = clf.fit(dataset_orig_train.features, dataset_orig_train.labels)
            pos_ind = np.where(clf.classes_ == dataset_orig_train.favorable_label)[0][0]
            train_pred = clf.predict(dataset_orig_train.features).reshape(-1, 1)
            train_prob = clf.predict_proba(dataset_orig_train.features)[:, pos_ind].reshape(-1, 1)
            pred = clf.predict(dataset_orig_test.features).reshape(-1, 1)
            pred_prob = clf.predict_proba(dataset_orig_test.features)[:, pos_ind].reshape(-1, 1)
        elif classifier_name in ["dl"]:
            clf.fit(dataset_orig_train.features, dataset_orig_train.labels, epochs=20)
            train_pred = (clf.predict(dataset_orig_train.features) > 0.5).reshape(-1, 1).astype("int32")
            train_prob = clf.predict(dataset_orig_train.features).reshape(-1, 1)
            pred = (clf.predict(dataset_orig_test.features) > 0.5).reshape(-1, 1).astype("int32")
            pred_prob = clf.predict(dataset_orig_test.features).reshape(-1, 1)

        dataset_orig_train_pred = dataset_orig_train.copy()
        dataset_orig_train_pred.labels = train_pred
        dataset_orig_train_pred.scores = train_prob

        dataset_orig_test_pred = dataset_orig_test.copy()
        dataset_orig_test_pred.labels = pred
        dataset_orig_test_pred.scores = pred_prob

        roc = RejectOptionClassification(privileged_groups=privileged_groups[dataset_name],
                                         unprivileged_groups=unprivileged_groups[dataset_name],
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                         num_class_thresh=100, num_ROC_margin=50,
                                         metric_name="Statistical parity difference",
                                         metric_ub=0.05, metric_lb=-0.05)
        roc = roc.fit(dataset_orig_train, dataset_orig_train_pred)
        pred_cpp = roc.predict(dataset_orig_test_pred)

        round_result = measure_final_score(dataset_orig_test, pred_cpp, sensitive_attributes)

        with open(osp.join(save_root, classifier_name + "_metrics.json"), "w") as f:
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
                    pool.apply_async(train_with_roc, args=(dataset, clf, [seed]))

        pool.close()
        pool.join()