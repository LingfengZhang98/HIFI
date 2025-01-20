"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/FSE22-MAAT/tree/main/MAAT
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Fair3602/maat.py

MAAT is an ensemble learning approach to improving fairness-performance trade-off for ML software
by simply combining models optimized for different objectives: fairness and ML performance.
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


def training_data_debugging(training_data, protected_attribute):
    """
    Perform undersampling by reducing samples in the Privileged & Favorable and Unprivileged & Unfavorable subgroups
    to alleviate the label bias as well as the selection bias.
    """
    zero_zero = len(
        training_data[(training_data['Probability'] == 0) & (training_data[protected_attribute] == 0)])
    zero_one = len(
        training_data[(training_data['Probability'] == 0) & (training_data[protected_attribute] == 1)])
    one_zero = len(
        training_data[(training_data['Probability'] == 1) & (training_data[protected_attribute] == 0)])
    one_one = len(
        training_data[(training_data['Probability'] == 1) & (training_data[protected_attribute] == 1)])

    a = zero_one+one_one
    b = -1*(zero_zero*zero_one+2*zero_zero*one_one+one_zero*one_one)
    c = (zero_zero+one_zero)*(zero_zero*one_one-zero_one*one_zero)
    x = (-b-math.sqrt(b*b-4*a*c))/(2*a)
    y = (zero_one+one_one)/(zero_zero+one_zero)*x

    zero_zero_new = int(zero_zero-x)
    one_one_new = int(one_one-y)

    zero_one_set = training_data[
        (training_data['Probability'] == 0) & (training_data[protected_attribute] == 1)]
    one_zero_set = training_data[
        (training_data['Probability'] == 1) & (training_data[protected_attribute] == 0)]

    if zero_zero_new < zero_zero:
        zero_zero_set = training_data[
            (training_data['Probability'] == 0) & (training_data[protected_attribute] == 0)].sample(
            n=zero_zero_new)
    else:
        zero_zero_set = training_data[(training_data['Probability'] == 0) & (training_data[protected_attribute] == 0)]

    if one_one_new < one_one:
        one_one_set = training_data[
            (training_data['Probability'] == 1) & (training_data[protected_attribute] == 1)].sample(
            n=one_one_new)
    else:
        one_one_set = training_data[(training_data['Probability'] == 1) & (training_data[protected_attribute] == 1)]

    new_set = pd.concat([zero_one_set, one_zero_set, zero_zero_set, one_one_set], ignore_index=True).sample(frac=1)

    return new_set


def train_with_maat(dataset_name, classifier_name, seed_range=[0]):
    method_name = "maat"
    sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
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

        datasets_undersampling = []
        for sen_attr in sensitive_attributes:
            dataset_wae = training_data_debugging(dataset_orig_train, sen_attr)
            datasets_undersampling.append(
                BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_wae,
                                                label_names=['Probability'],
                                                protected_attribute_names=[sen_attr]))

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                                label_names=['Probability'],
                                                protected_attribute_names=sensitive_attributes)
        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=sensitive_attributes)

        clf_suffix = copy.deepcopy(sensitive_attributes)
        clf_suffix.append("orig")
        datasets_undersampling.append(dataset_orig_train)
        pred_probs = []

        for i, suffix in enumerate(clf_suffix):
            clf = get_classifier(classifier_name, datasets_undersampling[i].features.shape[1:])
            if classifier_name == "dl":
                clf.fit(datasets_undersampling[i].features, datasets_undersampling[i].labels, epochs=20)
                pred_probs.append(np.array(clf.predict(dataset_orig_test.features)).reshape(-1,1))
                clf.save(osp.join(save_root, classifier_name+"_"+suffix+".keras"))
            else:
                if classifier_name == 'svm':
                    clf = CalibratedClassifierCV(estimator=clf)
                clf = clf.fit(datasets_undersampling[i].features, datasets_undersampling[i].labels)
                pred_probs.append(np.array(clf.predict_proba(dataset_orig_test.features)).reshape(-1,2))
                joblib.dump(clf, osp.join(save_root, classifier_name+"_"+suffix+".pkl"))

        if classifier_name != "dl":
            pred_probs = [x[:, 1].reshape(-1, 1) for x in pred_probs]
        pred_probs = np.hstack(tuple(pred_probs))
        prob_avg = np.mean(pred_probs, axis=1).reshape(-1, 1)
        ensemble_labels = np.where(prob_avg >= 0.5, 1, 0)

        test_df_copy = copy.deepcopy(dataset_orig_test)
        test_df_copy.labels = ensemble_labels

        round_result = measure_final_score(dataset_orig_test, test_df_copy, sensitive_attributes)

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
                    pool.apply_async(train_with_maat, args=(dataset, clf, [seed]))

        pool.close()
        pool.join()