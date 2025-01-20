"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/FairSMOTE2

Fair-SMOTE rebalances internal distributions on target class and sensitive attributes by oversampling,
and removes discriminatory samples in the context of individual fairness.
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

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import random

from aif360.datasets import BinaryLabelDataset

from tools.utils import set_seed, makedirs
from tools.config import SPLIT_RATIO, considered_sensitive_attributes
from tools.get_classifier import get_classifier
from tools.metrics import measure_final_score


def generate_group_class_combinations(length):
    combs = list(range(2 ** length))
    combs = [np.binary_repr(comb, width=length) for comb in combs]
    combs = [[int(item) for item in comb] for comb in combs]
    return combs


def get_ngbr(df, knn):
    rand_sample_idx = random.randint(0, df.shape[0] - 1)
    parent_candidate = df.iloc[rand_sample_idx]
    ngbr = knn.kneighbors(parent_candidate.values.reshape(1, -1), 3, return_distance=False)
    candidate_1 = df.iloc[ngbr[0][0]]
    candidate_2 = df.iloc[ngbr[0][1]]
    candidate_3 = df.iloc[ngbr[0][2]]
    return parent_candidate, candidate_2, candidate_3


def generate_samples(no_of_samples, df):
    total_data = df.values.tolist()
    knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(df)

    for _ in range(no_of_samples):
        cr = 0.8
        f = 0.8
        parent_candidate, child_candidate_1, child_candidate_2 = get_ngbr(df, knn)
        new_candidate = []
        for key, value in parent_candidate.items():
            if isinstance(parent_candidate[key], bool):
                new_candidate.append(parent_candidate[key] if cr < random.random() else not parent_candidate[key])
            elif isinstance(parent_candidate[key], str):
                new_candidate.append(
                    random.choice([parent_candidate[key], child_candidate_1[key], child_candidate_2[key]]))
            elif isinstance(parent_candidate[key], list):
                temp_lst = []
                for i, each in enumerate(parent_candidate[key]):
                    temp_lst.append(parent_candidate[key][i] if cr < random.random() else
                                    int(parent_candidate[key][i] +
                                        f * (child_candidate_1[key][i] - child_candidate_2[key][i])))
                new_candidate.append(temp_lst)
            else:
                new_candidate.append(abs(parent_candidate[key] + f * (child_candidate_1[key] - child_candidate_2[key])))
        total_data.append(new_candidate)

    final_df = pd.DataFrame(total_data)

    return final_df


def situation(clf, X_train, y_train, sensitive_attributes):
    columns_to_drop = []
    idx_rest = (X_train[sensitive_attributes[0]] == X_train[sensitive_attributes[0]])
    X_train_tmp = X_train.copy()
    for sen_attr in sensitive_attributes:
        X_flip = X_train.copy()
        X_flip[sen_attr] = np.where(X_flip[sen_attr] == 1, 0, 1)
        a = np.array(clf.predict(X_train))
        b = np.array(clf.predict(X_flip))
        same = (a == b)
        same = [1 if each else 0 for each in same]
        X_train_tmp['same_'+sen_attr] = same
        columns_to_drop.append('same_'+sen_attr)
        idx_rest = idx_rest & (X_train_tmp['same_'+sen_attr]==1)

    X_train_tmp['y'] = y_train
    X_rest = X_train_tmp[idx_rest]
    y_rest = X_rest['y']
    columns_to_drop.append('y')
    X_rest = X_rest.drop(columns=columns_to_drop)
    return X_rest.values, y_rest.values.reshape(-1, 1)


def train_with_fairsmote(dataset_name, classifier_name, seed_range=[0]):
    method_name = "fairsmote"
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

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']

        combs = generate_group_class_combinations(len(sensitive_attributes) + 1)
        len_group_class = []
        df_group_class = []
        for comb in combs:
            idx_group_class = dataset_orig_train['Probability'] == comb[-1]
            for i, sen_attr in enumerate(sensitive_attributes):
                idx_group_class = idx_group_class & (dataset_orig_train[sen_attr] == comb[i])
            len_group_class.append(len(dataset_orig_train[idx_group_class]))
            df_group_class.append(dataset_orig_train[idx_group_class])

        maximum = max(len_group_class)
        len_to_be_increased = [maximum - num for num in len_group_class]

        for dfgc in df_group_class:
            for sen_attr in sensitive_attributes:
                dfgc[sen_attr] = dfgc[sen_attr].astype(str)

        print("Start generating samples...")
        for i in range(len(combs)):
            df_group_class[i] = generate_samples(len_to_be_increased[i], df_group_class[i])
        df = pd.concat(df_group_class)

        df.columns = dataset_orig_train.columns
        for sen_attr in sensitive_attributes:
            df[sen_attr] = df[sen_attr].astype(float)
        clf_remove = RandomForestClassifier()
        clf_remove.fit(X_train, y_train)
        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
        print("Situational testing...")
        X_train, y_train = situation(clf_remove, X_train, y_train, sensitive_attributes)

        dataset_oversampled_train = pd.DataFrame(np.hstack((X_train, y_train)), columns=dataset_orig.columns)

        dataset_oversampled_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_oversampled_train,
                                                label_names=['Probability'],
                                                protected_attribute_names=sensitive_attributes)
        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=sensitive_attributes)

        clf = get_classifier(classifier_name, dataset_oversampled_train.features.shape[1:])
        if classifier_name == "dl":
            clf.fit(dataset_oversampled_train.features, dataset_oversampled_train.labels, epochs=20)
            preds = (clf.predict(dataset_orig_test.features) > 0.5).astype("int32")
            clf.save(osp.join(save_root, classifier_name+".keras"))
        else:
            clf = clf.fit(dataset_oversampled_train.features, dataset_oversampled_train.labels)
            preds = clf.predict(dataset_orig_test.features).reshape(-1, 1)
            joblib.dump(clf, osp.join(save_root, classifier_name+".pkl"))

        test_df_copy = copy.deepcopy(dataset_orig_test)
        test_df_copy.labels = preds

        round_result = measure_final_score(dataset_orig_test, test_df_copy, sensitive_attributes)

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
                    pool.apply_async(train_with_fairsmote, args=(dataset, clf, [seed]))

        pool.close()
        pool.join()