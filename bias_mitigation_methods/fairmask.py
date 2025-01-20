"""
This code is partially adapted from:
- https://github.com/anonymous12138/biasmitigation/blob/main/xFAIR/xFAIR.py
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/FairMask2/Fairmask.py

FairMask trains extrapolation models to predict protected attributes based on non-protected features,
and uses these extrapolation models to modify the protected attributes in test data,
enabling fairer predictions (in terms of both group fairness and individual fairness).
The original paper reported that the performance of FairMask is robust regardless of the choice of extrapolation model.
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

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from imblearn.over_sampling import SMOTE

from aif360.datasets import BinaryLabelDataset

from tools.utils import set_seed, makedirs
from tools.config import SPLIT_RATIO, considered_sensitive_attributes
from tools.get_classifier import get_classifier
from tools.metrics import measure_final_score


def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out


def train_with_fairmask(dataset_name, classifier_name, seed_range=[0]):
    method_name = "fairmask"
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

        X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
        y_train = copy.deepcopy(dataset_orig_train['Probability'])
        X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
        y_test = copy.deepcopy(dataset_orig_test['Probability'])

        if len(sensitive_attributes) == 1:
            attr = sensitive_attributes[0]

            reduced = list(X_train.columns)
            reduced.remove(attr)
            X_reduced, y_reduced = X_train.loc[:, reduced], X_train[attr]
            # Build model to predict the protect attribute
            clf1 = DecisionTreeRegressor()
            sm = SMOTE()
            X_trains, y_trains = sm.fit_resample(X_reduced, y_reduced)
            clf = get_classifier(classifier_name, X_trains.shape[1:])
            if classifier_name == 'svm':
                clf = CalibratedClassifierCV(estimator=clf)
            clf.fit(X_trains, y_trains)
            if classifier_name in ["lr", "rf", "svm"]:
                y_proba = clf.predict_proba(X_trains)
                y_proba = [each[1] for each in y_proba]
            elif classifier_name in ["dl"]:
                y_proba = clf.predict(X_trains)
            if isinstance(clf1, DecisionTreeClassifier) or isinstance(clf1, LogisticRegression):
                clf1.fit(X_trains, y_trains)
            else:
                clf1.fit(X_trains, y_proba)

            X_test_reduced = X_test.loc[:, X_test.columns != attr]
            protected_pred = clf1.predict(X_test_reduced)
            if isinstance(clf1, DecisionTreeRegressor) or isinstance(clf1, LinearRegression):
                protected_pred = reg2clf(protected_pred, threshold=0.5)

            # Build model to predict the target attribute Y
            clf2 = get_classifier(classifier_name, X_train.shape[1:])
            X_test.loc[:, attr] = protected_pred

            if classifier_name in ["lr", "rf", "svm"]:
                clf2.fit(X_train, y_train)
                y_pred = clf2.predict(X_test).reshape(-1, 1)
            elif classifier_name in ["dl"]:
                clf2.fit(X_train, y_train, epochs=20)
                y_pred = (clf2.predict(X_test) > 0.5).astype("int32")

        elif len(sensitive_attributes) == 2:
            reduced = list(X_train.columns)
            reduced.remove(sensitive_attributes[0])
            reduced.remove(sensitive_attributes[1])

            X_reduced, y_reduced0, y_reduced1 = X_train.loc[:, reduced], X_train[sensitive_attributes[0]], X_train[sensitive_attributes[1]]
            # Build model to predict the protect attribute0
            clf1_0 = DecisionTreeRegressor()
            sm = SMOTE()
            X_trains, y_trains0 = sm.fit_resample(X_reduced, y_reduced0)
            clf = get_classifier(classifier_name, X_trains.shape[1:])
            if classifier_name == 'svm':
                clf = CalibratedClassifierCV(estimator=clf)
            clf.fit(X_trains, y_trains0)
            if classifier_name in ["lr", "rf", "svm"]:
                y_proba = clf.predict_proba(X_trains)
                y_proba = [each[1] for each in y_proba]
            elif classifier_name in ["dl"]:
                y_proba = clf.predict(X_trains)
            if isinstance(clf1_0, DecisionTreeClassifier) or isinstance(clf1_0, LogisticRegression):
                clf1_0.fit(X_trains, y_trains0)
            else:
                clf1_0.fit(X_trains, y_proba)

            # Build model to predict the protect attribute1
            clf1_1 = DecisionTreeRegressor()
            sm = SMOTE()
            X_trains, y_trains1 = sm.fit_resample(X_reduced, y_reduced1)
            clf = get_classifier(classifier_name, X_trains.shape[1:])
            if classifier_name == 'svm':
                clf = CalibratedClassifierCV(estimator=clf)
            clf.fit(X_trains, y_trains1)
            if classifier_name in ["lr", "rf", "svm"]:
                y_proba = clf.predict_proba(X_trains)
                y_proba = [each[1] for each in y_proba]
            elif classifier_name in ["dl"]:
                y_proba = clf.predict(X_trains)
            if isinstance(clf1_1, DecisionTreeClassifier) or isinstance(clf1_1, LogisticRegression):
                clf1_1.fit(X_trains, y_trains0)
            else:
                clf1_1.fit(X_trains, y_proba)

            X_test_reduced = X_test.loc[:, reduced]
            protected_pred0 = clf1_0.predict(X_test_reduced)
            protected_pred1 = clf1_1.predict(X_test_reduced)
            if isinstance(clf1_0, DecisionTreeRegressor) or isinstance(clf1_0, LinearRegression):
                protected_pred0 = reg2clf(protected_pred0, threshold=0.5)
                protected_pred1 = reg2clf(protected_pred1, threshold=0.5)

            # Build model to predict the target attribute Y
            clf2 = get_classifier(classifier_name, X_train.shape[1:])
            X_test.loc[:, sensitive_attributes[0]] = protected_pred0
            X_test.loc[:, sensitive_attributes[1]] = protected_pred1

            if classifier_name in ["lr", "rf", "svm"]:
                clf2.fit(X_train, y_train)
                y_pred = clf2.predict(X_test).reshape(-1, 1)
            elif classifier_name in ["dl"]:
                clf2.fit(X_train, y_train, epochs=20)
                y_pred = (clf2.predict(X_test) > 0.5).astype("int32")

        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=sensitive_attributes)
        test_df_copy = copy.deepcopy(dataset_orig_test)
        test_df_copy.labels = y_pred

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
                    pool.apply_async(train_with_fairmask, args=(dataset, clf, [seed]))

        pool.close()
        pool.join()