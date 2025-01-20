"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/tree/main/Analysis_code

This experiment aims to comprehensively evaluate HIFI in terms of fairness improvement,
preserving model utilities, and fairness-performance trade-off. All experiments below
are based on the results of multiple repeated runs with different random seeds.
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from cliffs_delta import cliffs_delta
from scipy.stats import spearmanr, kendalltau

from aif360.datasets import BinaryLabelDataset

from tools.utils import makedirs, set_seed
from tools.fairea import fairea_classify_region, create_fairea_baseline
from tools.causal_fairness_evaluation import evaluate_causal_fairness
from tools.config import SPLIT_RATIO, considered_sensitive_attributes

from models.model_utils import load_original_model, model_standardize


def mann_p_value(x, y):
    """
    Mann Whitney U-test
    """
    return stats.mannwhitneyu(x, y)[1]

def cliffs_delta_value(x, y):
    """
    Cliff’s δ
    """
    return cliffs_delta(x, y)[0]


def load_orig_metrics(method_names, dataset_names, classifier_names, seed_range, save_root):
    metrics = {}
    for dataset in dataset_names:
        metrics[dataset] = {}
        for method_name in method_names:
            metrics[dataset][method_name] = {}
            for classifier_name in classifier_names:
                metrics[dataset][method_name][classifier_name] = {}

    for dataset in dataset_names:
        for method_name in method_names:
            if method_name in ["adv", "meta", "pr"]:
                clf_names = ["lr"]
            else:
                clf_names = classifier_names
            for clf_name in clf_names:
                if "hifi" in method_name and clf_name in ["rf"]:
                    continue
                for i, seed in enumerate(seed_range):
                    with open(osp.join(osp.dirname(__file__), "../models", dataset, method_name, "seed_"+str(seed), clf_name+"_metrics.json"),
                              "r", encoding="UTF-8") as f:
                        loaded_metric = json.loads(f.read())
                    if i == 0:
                        metric_names = list(loaded_metric.keys())
                        for metric_name in metric_names:
                            metrics[dataset][method_name][clf_name][metric_name] = []
                    for metric_name in metric_names:
                        if isinstance(loaded_metric[metric_name], list):
                            metrics[dataset][method_name][clf_name][metric_name].append(loaded_metric[metric_name][-1])
                        else:
                            metrics[dataset][method_name][clf_name][metric_name].append(loaded_metric[metric_name])

    makedirs(save_root)
    with open(osp.join(save_root, "aggregated_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)


def calculate_default_statistics(dataset_names, classifier_names, save_root):
    with open(osp.join(save_root, "aggregated_metrics.json"), "r", encoding="UTF-8") as f:
        metrics = json.loads(f.read())
    fout = open(osp.join(save_root, "original_performance.txt"), "w")

    fout.write("The original performance averaged on all model types:\n")
    for dataset in dataset_names:
        fout.write(dataset + ":\n")
        header_metric = list(metrics[dataset]["default"][classifier_names[0]].keys())
        for metric_name in header_metric:
            fout.write(metric_name + "\t")
        fout.write("\n")
        for metric_name in header_metric:
            value_list = []
            for clf_name in classifier_names:
                if not metrics[dataset]["default"][clf_name]:
                    continue
                else:
                    value_list.append(np.mean(metrics[dataset]["default"][clf_name][metric_name]))
            fout.write("%.3f\t" % (np.mean(value_list)))
        fout.write("\n")
    fout.close()


def calculate_average_changes(method_names, dataset_names, classifier_names, save_root, model_wise_root=None, hifi_evaluation=False):
    if hifi_evaluation:
        with open(osp.join(save_root, "hifi_aggregated_metrics.json"), "r", encoding="UTF-8") as f:
            metrics = json.loads(f.read())
        if model_wise_root is None:
            fout = open(osp.join(save_root, "hifi_metrics_average_changes.txt"), "w")
        else:
            fout = open(osp.join(save_root, model_wise_root, "hifi_metrics_average_changes.txt"), "w")
    else:
        with open(osp.join(save_root, "aggregated_metrics.json"), "r", encoding="UTF-8") as f:
            metrics = json.loads(f.read())
        if model_wise_root is None:
            fout = open(osp.join(save_root, "metrics_average_changes.txt"), "w")
        else:
            fout = open(osp.join(save_root, model_wise_root, "metrics_average_changes.txt"), "w")

    fout.write("The absolute and relative changes on each dataset:\n")
    for dataset in dataset_names:
        fout.write(dataset+":\n")
        header_metric = list(metrics[dataset]["default"][classifier_names[0]].keys())
        for metric_name in header_metric:
            fout.write("\t"+metric_name)
        fout.write("\n")
        for method_name in method_names:
            if method_name != "default":
                fout.write(method_name)
                absolute_change_list = {}
                base_value_list = {}
                for metric_name in header_metric:
                    absolute_change_list[metric_name] = []
                    base_value_list[metric_name] = []
                    for clf_name in classifier_names:
                        if not metrics[dataset][method_name][clf_name]:
                            continue
                        else:
                            absolute_change_list[metric_name].append(np.mean(metrics[dataset][method_name][clf_name][metric_name]) -
                                                                     np.mean(metrics[dataset]["default"][clf_name][metric_name]))
                            base_value_list[metric_name].append(np.mean(metrics[dataset]["default"][clf_name][metric_name]))
                    fout.write("\t%.3f (%.1f" % (np.mean(absolute_change_list[metric_name]), (100*np.mean(absolute_change_list[metric_name])/np.mean(base_value_list[metric_name]))))
                    fout.write("\\%)")
                fout.write("\n")
    fout.write("\n")

    fout.write("The absolute and relative changes on each model type:\n")
    for clf_name in classifier_names:
        fout.write(clf_name + ":\n")
        header_metric = list(metrics[dataset_names[0]]["default"][classifier_names[0]].keys())[:5]
        header_metric = header_metric + ["SPD", "AOD", "EOD"]
        for metric_name in header_metric:
            fout.write("\t"+metric_name)
        fout.write("\n")
        for method_name in method_names:
            if method_name != "default":
                fout.write(method_name)
                if ((clf_name != "lr") & (method_name in ["adv", "meta", "pr"])) | ((clf_name not in ["lr", "dl", "svm"]) & ("hifi" in method_name)):
                    fout.write("\n")
                    continue
                absolute_change_list = {}
                base_value_list = {}
                for i, metric_name in enumerate(header_metric):
                    absolute_change_list[metric_name] = []
                    base_value_list[metric_name] = []
                    for dataset in dataset_names:
                        header_metric_orig_fairness = list(metrics[dataset]["default"][clf_name].keys())[5:]
                        if i < 5:   # for performance metrics
                            absolute_change_list[metric_name].append(
                                np.mean(metrics[dataset][method_name][clf_name][metric_name]) -
                                np.mean(metrics[dataset]["default"][clf_name][metric_name]))
                            base_value_list[metric_name].append(np.mean(metrics[dataset]["default"][clf_name][metric_name]))
                        else:   # for fairness metrics, take average by the fairness definitions (e.g., "SPD", "AOD", "EOD")
                            for fairness_metric in header_metric_orig_fairness:
                                if fairness_metric.endswith(metric_name):
                                    absolute_change_list[metric_name].append(
                                        np.mean(metrics[dataset][method_name][clf_name][fairness_metric]) -
                                        np.mean(metrics[dataset]["default"][clf_name][fairness_metric]))
                                    base_value_list[metric_name].append(np.mean(metrics[dataset]["default"][clf_name][fairness_metric]))
                    fout.write("\t%.3f (%.1f" % (np.mean(absolute_change_list[metric_name]), (100*np.mean(absolute_change_list[metric_name])/np.mean(base_value_list[metric_name]))))
                    fout.write("\\%)")
                fout.write("\n")
    fout.write("\n")

    fout.write("The absolute metrics and relative changes averaged on all datasets and model types:\n")
    header_metric = list(metrics[dataset_names[0]]["default"][classifier_names[0]].keys())[:5]
    header_metric = header_metric + ["SPD", "AOD", "EOD"]
    for metric_name in header_metric:
        fout.write("\t" + metric_name)
    fout.write("\n")
    base_value_list = {}
    for method_name in method_names:
        fout.write(method_name)
        fairness_method_value_list = {}
        for i, metric_name in enumerate(header_metric):
            fairness_method_value_list[metric_name] = []
            for dataset in dataset_names:
                header_metric_orig_fairness = list(metrics[dataset]["default"][classifier_names[0]].keys())[5:]
                for clf_name in classifier_names:
                    if not metrics[dataset][method_name][clf_name]:
                        continue
                    else:
                        if i < 5:  # for performance metrics
                            fairness_method_value_list[metric_name].append(
                                np.mean(metrics[dataset][method_name][clf_name][metric_name]))
                        else:  # for fairness metrics, take average by the fairness definitions (e.g., "SPD", "AOD", "EOD")
                            for fairness_metric in header_metric_orig_fairness:
                                if fairness_metric.endswith(metric_name):
                                    fairness_method_value_list[metric_name].append(
                                        np.mean(metrics[dataset][method_name][clf_name][fairness_metric]))
            absolute_value = np.mean(fairness_method_value_list[metric_name])
            if method_name == "default":
                base_value_list[metric_name] = absolute_value
            relative_change = 100.0 * (absolute_value - base_value_list[metric_name]) / base_value_list[metric_name]
            fout.write("\t%.3f (%.1f" % (absolute_value, relative_change))
            fout.write("\\%)")
        fout.write("\n")
    fout.close()


def statistical_test(method_names, dataset_names, classifier_names, save_root, results_save_root=None, hifi_evaluation=False):
    if hifi_evaluation:
        with open(osp.join(save_root, "hifi_aggregated_metrics.json"), "r", encoding="UTF-8") as f:
            metrics = json.loads(f.read())
        if results_save_root is None:
            fout = open(osp.join(save_root, "hifi_scenario_statistics.txt"), "w")
        else:
            fout = open(osp.join(results_save_root, "hifi_scenario_statistics.txt"), "w")
    else:
        with open(osp.join(save_root, "aggregated_metrics.json"), "r", encoding="UTF-8") as f:
            metrics = json.loads(f.read())
        if results_save_root is None:
            fout = open(osp.join(save_root, "scenario_statistics.txt"), "w")
        else:
            fout = open(osp.join(results_save_root, "scenario_statistics.txt"), "w")

    fout.write("The proportions of scenarios where the methods improve fairness or degrade performance and where they also have significantly large effects:\n")
    fout.write("\tperformance_decrease\tperformance_decrease_large_effect\tperformance_increase_large_effect\tfairness_increase\tfairness_increase_large_effect\tfairness_decrease_large_effect\n")
    for method_name in method_names:
        if method_name != "default":
            fout.write(method_name)
            count_scenarios = {}
            count_scenarios["acc_total"] = 0
            count_scenarios["acc_decrease"] = 0
            count_scenarios["acc_sig_decrease"] = 0
            count_scenarios["acc_sig_increase"] = 0
            count_scenarios["fair_total"] = 0
            count_scenarios["fair_increase"] = 0
            count_scenarios["fair_sig_increase"] = 0
            count_scenarios["fair_sig_decrease"] = 0
            for dataset in dataset_names:
                header_metric = list(metrics[dataset]["default"][classifier_names[0]].keys())
                for clf_name in classifier_names:
                    if not metrics[dataset][method_name][clf_name]:
                        continue
                    else:
                        for i, metric_name in enumerate(header_metric):
                            if i < 5:  # for performance metrics
                                count_scenarios["acc_total"] += 1
                                if np.mean(metrics[dataset][method_name][clf_name][metric_name]) < np.mean(metrics[dataset]["default"][clf_name][metric_name]):
                                    count_scenarios["acc_decrease"] += 1
                                    if (mann_p_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name]) < 0.05) & (abs(cliffs_delta_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name])) >= 0.428):
                                        count_scenarios["acc_sig_decrease"] += 1
                                else:
                                    if (mann_p_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name]) < 0.05) & (abs(cliffs_delta_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name])) >= 0.428):
                                        count_scenarios["acc_sig_increase"] += 1
                            else:   # for fairness metrics
                                count_scenarios["fair_total"] += 1
                                if np.mean(metrics[dataset][method_name][clf_name][metric_name]) < np.mean(metrics[dataset]["default"][clf_name][metric_name]):
                                    count_scenarios["fair_increase"] += 1
                                    if (mann_p_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name]) < 0.05) & (abs(cliffs_delta_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name])) >= 0.428):
                                        count_scenarios["fair_sig_increase"] += 1
                                else:
                                    if (mann_p_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name]) < 0.05) & (abs(cliffs_delta_value(metrics[dataset][method_name][clf_name][metric_name], metrics[dataset]["default"][clf_name][metric_name])) >= 0.428):
                                        count_scenarios["fair_sig_decrease"] += 1
            fout.write("\t%.1f" % (100*count_scenarios["acc_decrease"]/count_scenarios["acc_total"]))
            fout.write("\\%")
            fout.write("\t%.1f" % (100*count_scenarios["acc_sig_decrease"]/count_scenarios["acc_total"]))
            fout.write("\\%")
            fout.write("\t%.1f" % (100 * count_scenarios["acc_sig_increase"] / count_scenarios["acc_total"]))
            fout.write("\\%")
            fout.write("\t%.1f" % (100*count_scenarios["fair_increase"]/count_scenarios["fair_total"]))
            fout.write("\\%")
            fout.write("\t%.1f" % (100*count_scenarios["fair_sig_increase"]/count_scenarios["fair_total"]))
            fout.write("\\%")
            fout.write("\t%.1f" % (100 * count_scenarios["fair_sig_decrease"] / count_scenarios["fair_total"]))
            fout.write("\\%\n")
    fout.close()


def hifi_load_orig_metrics(eta_list, dataset_names, classifier_names, seed_range, save_root, method_list = ["hifi"]):
    metrics = {}
    method_names = []

    # aggregate the methods from the original models
    method_name = "default"
    method_names.append(method_name)
    for dataset in dataset_names:
        metrics[dataset] = {}
        metrics[dataset][method_name] = {}
        for classifier_name in classifier_names:
            metrics[dataset][method_name][classifier_name] = {}
            for i, seed in enumerate(seed_range):
                with open(osp.join(osp.dirname(__file__), "../models", dataset, method_name, "seed_" + str(seed),
                                   classifier_name + "_metrics.json"),
                          "r", encoding="UTF-8") as f:
                    loaded_metric = json.loads(f.read())
                if i == 0:
                    metric_names = list(loaded_metric.keys())
                    for metric_name in metric_names:
                        metrics[dataset][method_name][classifier_name][metric_name] = []
                for metric_name in metric_names:
                    if isinstance(loaded_metric[metric_name], list):
                        metrics[dataset][method_name][classifier_name][metric_name].append(loaded_metric[metric_name][-1])
                    else:
                        metrics[dataset][method_name][classifier_name][metric_name].append(loaded_metric[metric_name])

    # aggregate the methods from our method HIFI with different eta
    for method_name in method_list:
        for eta in eta_list:
            method_version = method_name + f"_eta={eta}"
            method_names.append(method_version)
            for dataset in dataset_names:
                metrics[dataset][method_version] = {}
                for clf_name in classifier_names:
                    metrics[dataset][method_version][clf_name] = {}
                    if clf_name not in ["lr", "dl", "svm"]:
                        continue
                    for i, seed in enumerate(seed_range):
                        with open(osp.join(osp.dirname(__file__), "../models", dataset, method_name, "seed_"+str(seed), f"eta={eta}", clf_name+"_metrics.json"),
                                  "r", encoding="UTF-8") as f:
                            loaded_metric = json.loads(f.read())
                        if i == 0:
                            metric_names = list(loaded_metric.keys())
                            for metric_name in metric_names:
                                metrics[dataset][method_version][clf_name][metric_name] = []
                        for metric_name in metric_names:
                            if isinstance(loaded_metric[metric_name], list):
                                metrics[dataset][method_version][clf_name][metric_name].append(loaded_metric[metric_name][-1])
                            else:
                                metrics[dataset][method_version][clf_name][metric_name].append(loaded_metric[metric_name])

    makedirs(save_root)
    with open(osp.join(save_root, "hifi_aggregated_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    return method_names


def fairea_evaluation(method_names, dataset_names, classifier_names, seed_range, repetitions, degrees, save_root, results_save_root=None, hifi_evaluation=False):
    region_types = ["win-win", "good", "inverted", "poor", "lose-lose"]

    create_fairea_baseline(dataset_names, classifier_names, seed_range, repetitions, degrees)
    with open(osp.join(save_root, "fairea_baselines.json"), "r", encoding="UTF-8") as f:
        fairea_baselines = json.loads(f.read())

    if hifi_evaluation:
        with open(osp.join(save_root, "hifi_aggregated_metrics.json"), "r", encoding="UTF-8") as f:
            metrics = json.loads(f.read())
        if results_save_root is None:
            fout = open(osp.join(save_root, "hifi_fairea_evaluation.txt"), "w")
        else:
            fout = open(osp.join(results_save_root, "hifi_fairea_evaluation.txt"), "w")
    else:
        with open(osp.join(save_root, "aggregated_metrics.json"), "r", encoding="UTF-8") as f:
            metrics = json.loads(f.read())
        if results_save_root is None:
            fout = open(osp.join(save_root, "fairea_evaluation.txt"), "w")
        else:
            fout = open(osp.join(results_save_root, "fairea_evaluation.txt"), "w")

    fout.write("The ratio of mitigation regions of each bias mitigation method:\n")
    for region_type in region_types:
        fout.write(f"\t{region_type}")

    for method_name in method_names:
        fout.write(f"\n{method_name}")
        region_count = {}
        total_count = 0
        for region_type in region_types:
            region_count[region_type] = 0
        for dataset_name in dataset_names:
            header_metric = list(metrics[dataset_name]["default"][classifier_names[0]].keys())
            performance_names = header_metric[:5]
            fairness_names = header_metric[5:]
            for classifier_name in classifier_names:
                if not metrics[dataset_name][method_name][classifier_name]:
                    continue
                for performance_name in performance_names:
                    for fairness_name in fairness_names:
                        for i, seed in enumerate(seed_range):
                            region = fairea_classify_region(fairea_baselines[dataset_name][classifier_name][performance_name],
                                                            fairea_baselines[dataset_name][classifier_name][fairness_name],
                                                            metrics[dataset_name][method_name][classifier_name][performance_name][i],
                                                            metrics[dataset_name][method_name][classifier_name][fairness_name][i])
                            region_count[region] += 1
                            total_count += 1
        for region_type in region_types:
            if total_count != 0:
                fout.write("\t%.1f" % (region_count[region_type] / total_count * 100))
                fout.write("\\%")
            else:
                fout.write("\tX")
    fout.close()


def individual_fairness_statistics(method_names, dataset_names, classifier_names, seed_range):
    """
    Blue line in Fig.9 of the paper.
    """
    save_root = osp.join(osp.dirname(__file__), "../results/performance_fairness_analysis")
    fout = open(osp.join(save_root, "individual_discrimination_rate.txt"), "w")
    results = {}
    for method_name in method_names:
        results[method_name] = []
        fout.write("%s\t" % (method_name))
        for dataset_name in dataset_names:
            sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
            indices_sensitive_attributes = list(considered_sensitive_attributes[dataset_name].values())

            for classifier_name in classifier_names:
                if method_name in ["meta", "pr", "adv"] and classifier_name in ["svm", "rf", "dl"]:
                    continue
                elif "hifi" in method_name and classifier_name in ["rf"]:
                    continue
                else:
                    for seed in seed_range:
                        set_seed(seed)
                        data_root = osp.join(osp.dirname(__file__), "../data/tabular", dataset_name, dataset_name + "_processed.csv")
                        dataset_orig = pd.read_csv(data_root)
                        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=SPLIT_RATIO[dataset_name], shuffle=True)
                        scaler = MinMaxScaler()
                        scaler.fit(dataset_orig_train)
                        dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)
                        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0,
                                                               df=dataset_orig_test,
                                                               label_names=['Probability'],
                                                               protected_attribute_names=sensitive_attributes)
                        orig_model = load_original_model(method_name, dataset_name, classifier_name, seed)
                        model, _ = model_standardize(method_name, dataset_name, classifier_name, orig_model)
                        results[method_name].append(evaluate_causal_fairness(model, dataset_orig_test, indices_sensitive_attributes))
        fout.write("%.1f" % (np.mean(results[method_name]) * 100))
        fout.write("\\%\n")

    fout.close()


def correlation_between_individual_bias_and_purely_sensitive_interaction_strength(interaction_save_root, fairness_save_root):
    """
    Fig.9 in the paper.
    """
    list_avg_individual_discrimination_rate = []

    with open(osp.join(interaction_save_root, "avg_abs_purely_sensitive_interaction_values.txt"), "r") as f:
        lines = f.readlines()
        str_values = lines[2].strip().split()
        list_avg_abs_purely_sensitive_interaction = [float(value) for value in str_values]
    with open(osp.join(fairness_save_root, "individual_discrimination_rate.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                percentage_value = float(parts[1].strip("\%")) / 100
                list_avg_individual_discrimination_rate.append(percentage_value)

    fout = open(osp.join(fairness_save_root, "correlation_individual_fairness_purely_sensitive_interaction.txt"), "w")
    corr_statistic_methods = [("Spearman correlation coefficients", spearmanr),
                              ("Kendall's tau correlation coefficients", kendalltau)]
    for corr_statistic_method in corr_statistic_methods:
        fout.write("The %s between average strength of purely sensitive interactions, "
                   "and individual discrimination rate:\n" % (corr_statistic_method[0]))
        corr_func = corr_statistic_method[1]
        corr = corr_func(list_avg_abs_purely_sensitive_interaction, list_avg_individual_discrimination_rate)
        if corr.pvalue < 0.05:
            fout.write("\t%.3f*\n" % (corr.statistic))
        else:
            fout.write("\t%.3f\n" % (corr.statistic))
    fout.close()


if __name__ == '__main__':
    method_names = ["default", "rew", "fairsmote", "maat", "adv", "meta", "pr", "dir", "fairmask", "eop", "ceo", "roc"]
    dataset_names = ["census", "ufrgs", "compas", "diabetes", "default"]
    classifier_names = ["dl", "lr", "rf", "svm"]
    seed_range = list(range(0, 10))
    repetitions = 10
    degrees = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    save_root = osp.join(osp.dirname(__file__), "../results", "performance_fairness_analysis")
    interaction_analysis_save_root = osp.join(osp.dirname(__file__), "../results", "harsanyi_interaction_analysis")

    load_orig_metrics(method_names, dataset_names, classifier_names, seed_range, save_root)
    calculate_default_statistics(dataset_names, classifier_names, save_root)

    eta_list = [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 100, 1000]
    hifi_method_list = ["hifi", "rew+hifi", "maat+hifi", "fairmask+hifi"]
    hifi_method_names = hifi_load_orig_metrics(eta_list, dataset_names, classifier_names, seed_range, save_root, hifi_method_list)

    # Table V
    calculate_average_changes(hifi_method_names, dataset_names, classifier_names, save_root, hifi_evaluation=True)

    # Table. VI and Fig. 7
    calculate_average_changes(method_names, dataset_names, ["lr"], save_root, model_wise_root="model_wise/lr")
    calculate_average_changes(hifi_method_names, dataset_names, ["lr"], save_root, model_wise_root="model_wise/lr", hifi_evaluation=True)
    # Fig. 11
    calculate_average_changes(method_names, dataset_names, ["dl"], save_root, model_wise_root="model_wise/dl")
    calculate_average_changes(hifi_method_names, dataset_names, ["dl"], save_root, model_wise_root="model_wise/dl", hifi_evaluation=True)

    # blue line in Fig. 9, and correlation analysis
    method_names_fig9 = ["default", "fairsmote", "pr", "maat", "rew", "adv", "hifi_eta=0.75"]
    individual_fairness_statistics(method_names_fig9, dataset_names, classifier_names, seed_range)
    correlation_between_individual_bias_and_purely_sensitive_interaction_strength(interaction_analysis_save_root, save_root)

    hifi_to_be_evaluated = ["hifi_eta=0.75"]
    classifier_names_dw = ["lr"]
    # dataset-wise evaluation in Fig. 8a
    for dataset_name in dataset_names:
        save_root_dataset_wise = osp.join(save_root, "dataset_wise", dataset_name)
        makedirs(save_root_dataset_wise)
        statistical_test(method_names, [dataset_name], classifier_names_dw, save_root, save_root_dataset_wise)
        statistical_test(hifi_to_be_evaluated, [dataset_name], classifier_names_dw, save_root, save_root_dataset_wise, hifi_evaluation=True)
        fairea_evaluation(method_names, [dataset_name], classifier_names_dw, seed_range, repetitions, degrees, save_root, save_root_dataset_wise)
        fairea_evaluation(hifi_to_be_evaluated, [dataset_name], classifier_names_dw, seed_range, repetitions, degrees, save_root, save_root_dataset_wise, hifi_evaluation=True)

    # model-wise evaluation in Fig. 4, 5, 6, and 8b
    for classifier_name in classifier_names:
        save_root_dataset_wise = osp.join(save_root, "model_wise", classifier_name)
        makedirs(save_root_dataset_wise)
        statistical_test(method_names, dataset_names, [classifier_name], save_root, save_root_dataset_wise)
        statistical_test(hifi_to_be_evaluated, dataset_names, [classifier_name], save_root, save_root_dataset_wise, hifi_evaluation=True)
        fairea_evaluation(method_names, dataset_names, [classifier_name], seed_range, repetitions, degrees, save_root, save_root_dataset_wise)
        fairea_evaluation(hifi_to_be_evaluated, dataset_names, [classifier_name], seed_range, repetitions, degrees, save_root, save_root_dataset_wise, hifi_evaluation=True)