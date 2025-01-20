"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Fairea_multi/fairea.py
- https://github.com/maxhort/Fairea/blob/main/Fairea/fairea.py

Fairea measures the fairness-accuracy trade-off of ML bias mitigation methods with a novel baseline
constructed by model behavior mutation.
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import json
import warnings
warnings.filterwarnings('ignore')

from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from shapely.geometry import LineString, Point

from aif360.datasets import BinaryLabelDataset

from tools.utils import set_seed
from tools.config import SPLIT_RATIO, considered_sensitive_attributes, majority_label
from tools.metrics import measure_final_score

from models.model_utils import load_original_model, model_standardize


def create_fairea_baseline(dataset_names, classifier_names, seed_range, repetitions, degrees):
    fairea_baseline = {}
    orig_method = "default"
    scaler = MinMaxScaler()

    for dataset_name in dataset_names:
        fairea_baseline[dataset_name] = {}
        sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
        dataset_orig = pd.read_csv(osp.join(osp.dirname(__file__), "../data/tabular", dataset_name, dataset_name + "_processed.csv"))

        for classifier_name in classifier_names:
            fairea_baseline[dataset_name][classifier_name] = {}
            results = defaultdict(lambda: defaultdict(list))

            for seed in seed_range:
                set_seed(seed)

                dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=SPLIT_RATIO[dataset_name], shuffle=True)
                scaler.fit(dataset_orig_train)
                dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)
                dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                                       label_names=['Probability'],
                                                       protected_attribute_names=sensitive_attributes)

                num_test = len(dataset_orig_test.labels)
                idx_list = list(range(num_test))

                orig_model = load_original_model(orig_method, dataset_name, classifier_name, seed)
                clf, _ = model_standardize(orig_method, dataset_name, classifier_name, orig_model)

                pred = (clf(dataset_orig_test.features) > 0.5).astype("int32")
                dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
                dataset_orig_test_pred.labels = pred

                # Mutate labels for each degree
                for degree in degrees:
                    # total number of labels to mutate
                    num_mutate = int(num_test * degree)

                    for i in range(repetitions):
                        set_seed(i)
                        to_mutate = np.random.choice(idx_list, size=num_mutate, replace=False)
                        mutated_labels = np.copy(pred)
                        mutated_labels[to_mutate] = np.full((num_mutate, 1), majority_label[dataset_name])
                        dataset_orig_test_pred.labels = mutated_labels
                        round_results = measure_final_score(dataset_orig_test, dataset_orig_test_pred, sensitive_attributes)
                        header_metric = list(round_results.keys())

                        for j, metric in enumerate(header_metric):
                            if j < 5:   # performance
                                results[metric][degree].append(round_results[metric])
                            else:       # fairness
                                results[metric][degree].append(round_results[metric][-1])

            for metric in header_metric:
                fairea_baseline[dataset_name][classifier_name][metric] = []
                for degree in degrees:
                    fairea_baseline[dataset_name][classifier_name][metric].append(np.mean(np.array(results[metric][degree])))

    with open(osp.join(osp.dirname(__file__), "../results/performance_fairness_analysis", "fairea_baselines.json"), "w") as f:
        json.dump(fairea_baseline, f, indent=1)


def fairea_classify_region(performance_baselines, fairness_baselines, performance_to_evaluate, fairness_to_evaluate):
    # normalize metrics
    performance_range = np.max(performance_baselines) - np.min(performance_baselines)
    fairness_range = np.max(fairness_baselines) - np.min(fairness_baselines)
    min_performance = np.min(performance_baselines)
    min_fairness = np.min(fairness_baselines)
    norm_performance_baselines = (performance_baselines - min_performance) / performance_range
    norm_fairness_baselines = (fairness_baselines - min_fairness) / fairness_range
    norm_performance_to_evaluate = (performance_to_evaluate - min_performance) / performance_range
    norm_fairness_to_evaluate = (fairness_to_evaluate - min_fairness) / fairness_range

    # determine bias mitigation region of normalized bias mitigation methods
    baseline = LineString([(x, y) for x, y in zip(norm_fairness_baselines, norm_performance_baselines)])
    p = Point(norm_fairness_to_evaluate, norm_performance_to_evaluate)
    # Extend bias mitigation point towards four directions (left,right,up,down)
    line_down = LineString([(p.x, p.y), (p.x, 0)])
    line_right = LineString([(p.x, p.y), (2, p.y)])
    line_up = LineString([(p.x, p.y), (p.x, 2)])
    line_left = LineString([(p.x, p.y), (0, p.y)])
    # Determine bias mitigation region based on intersection with baseline
    if baseline.intersects(line_down) and baseline.intersects(line_right):
        return "good"
    elif baseline.intersects(line_down):
        return "win-win"
    elif baseline.intersects(line_up):
        return "poor"
    elif baseline.intersects(line_left):
        return "lose-lose"
    elif norm_performance_to_evaluate < 0:
        return "lose-lose"
    else:
        return "inverted"


if __name__ == '__main__':
    dataset_names = ["census", "ufrgs", "compas", "diabetes", "default"]   # "census", "ufrgs", "compas", "diabetes", "heart", "default", "meps15", "meps16", "bank"
    classifier_names = ["dl", "lr", "rf", "svm"]
    seed_range = list(range(0, 10))
    degrees = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    create_fairea_baseline(dataset_names, classifier_names, seed_range, 10, degrees)