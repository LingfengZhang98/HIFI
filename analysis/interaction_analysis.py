"""
This experiment aims to preliminarily figure out what kinds of Harsanyi interaction encode sensitive information,
and how the existing bias mitigation methods debias through the lens of game-theoretic interactions.
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import warnings
warnings.filterwarnings('ignore')

import multiprocessing
import json
import random
import itertools
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, kendalltau

from tools.utils import makedirs, set_seed, get_all_nonempty_subsets, get_all_subsets
from tools.config import SPLIT_RATIO, considered_sensitive_attributes, USED_SAMPLE_NUM
from harsanyi.and_harsanyi import AndHarsanyi
from harsanyi.harsanyi_utils import mask_input_fn_tabular
from models.model_utils import load_original_model, model_standardize


def calculate_interactions_on_sampled_data(method_name, dataset_name, classifier_name, seed):
    """
    Calculate the interactions and corresponding masks for randomly sampled data on each model.
    """
    data_root = osp.join(osp.dirname(__file__), "../data/tabular", dataset_name, dataset_name+"_processed.csv")
    result_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis", dataset_name, method_name, "seed_"+str(seed), classifier_name)
    makedirs(result_root)

    # calculate mean values on training set as the baseline, and sample data from testing set for experiments
    set_seed(seed)
    dataset_orig = pd.read_csv(data_root)
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=SPLIT_RATIO[dataset_name], shuffle=True)
    scaler = MinMaxScaler()
    scaler.fit(dataset_orig_train)
    dataset_orig_train = scaler.transform(dataset_orig_train)
    dataset_orig_test = scaler.transform(dataset_orig_test)
    mean_baseline = np.mean(dataset_orig_train[:, :-1], axis=0).reshape(1, -1)
    np.save(osp.join(result_root, "baseline.npy"), mean_baseline)
    idx_lst = list(range(dataset_orig_test.shape[0]))
    random.shuffle(idx_lst)
    X_test_sampled = dataset_orig_test[idx_lst][:USED_SAMPLE_NUM[dataset_name], :-1]
    y_test_sampled = dataset_orig_test[idx_lst][:USED_SAMPLE_NUM[dataset_name], [-1]]
    np.save(osp.join(result_root, "X_test_sampled.npy"), X_test_sampled)
    np.save(osp.join(result_root, "y_test_sampled.npy"), y_test_sampled)

    # load model
    orig_model = load_original_model(method_name, dataset_name, classifier_name, seed)
    model, reward_type = model_standardize(method_name, dataset_name, classifier_name, orig_model)

    # calculate Harsanyi interactions on sampled data
    masks = np.array([])
    list_interactions = []
    list_rewards = []
    for i in range(len(X_test_sampled)):
        print(f"Processing the sample [{i}] on {dataset_name}-{method_name}-{seed}-{classifier_name} \n")
        if method_name not in ["pr"]:
            calculator = AndHarsanyi(
                model=model,
                reward_type=reward_type,
                x=X_test_sampled[i].reshape(1, -1),
                y=y_test_sampled[i],
                baseline=mean_baseline,
                all_players=list(range(X_test_sampled.shape[1])),
                mask_input_fn=mask_input_fn_tabular
            )
        else:
            calculator = AndHarsanyi(
                model=model,
                reward_type=reward_type,
                x=X_test_sampled[i].reshape(1, -1),
                y=y_test_sampled[i],
                baseline=mean_baseline,
                all_players=list(range(X_test_sampled.shape[1])),
                mask_input_fn=mask_input_fn_tabular,
                verbose=1
            )
        calculator.attribute()
        if i == 0:
            masks = calculator.get_masks()
        interactions = calculator.get_interaction()
        rewards = calculator.get_rewards()
        list_interactions.append(interactions)
        list_rewards.append(rewards)
    np.save(osp.join(result_root, "masks.npy"), masks)
    np.save(osp.join(result_root, "interactions.npy"), np.array(list_interactions))
    np.save(osp.join(result_root, "rewards.npy"), np.array(list_rewards))


def fairness_gap_interaction_analysis(method_names, dataset_names, classifier_names, seed_range, fairness_metric):
    """
    Decompose group fairness in terms of SPD, AOD and EOD, into order-wise interactions.
    """
    save_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis")
    middle_points = ["0.1n", "0.3n", "0.5n", "0.7n", "0.9n"]
    interaction_types = ["sa", "non-sa", "all"]     # interactions involving sensitive attributes or not, and all interactions
    results = {}
    for method_name in method_names:
        results[method_name] = {}
        for interaction_type in interaction_types:
            results[method_name][interaction_type] = {}
            for middle_point in middle_points:
                results[method_name][interaction_type][middle_point] = []

    for method_name in method_names:
        for dataset_name in dataset_names:
            for classifier_name in classifier_names:
                if method_name in ["meta", "pr", "adv"] and classifier_name in ["svm", "rf", "dl"]:
                    continue
                else:
                    for seed in seed_range:
                        interaction_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis",
                                               dataset_name, method_name, "seed_" + str(seed), classifier_name)
                        masks = np.load(osp.join(interaction_root, "masks.npy"))
                        interactions = np.load(osp.join(interaction_root, "interactions.npy"))
                        interactions = np.squeeze(interactions, axis=-1)
                        X_test_sampled = np.load(osp.join(interaction_root, "X_test_sampled.npy"))
                        y_test_sampled = np.load(osp.join(interaction_root, "y_test_sampled.npy"))
                        sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
                        num_features = X_test_sampled.shape[-1]
                        interval_orders = {"0.1n": (0, 0.2 * num_features),
                                           "0.3n": (0.2 * num_features, 0.4 * num_features),
                                           "0.5n": (0.4 * num_features, 0.6 * num_features),
                                           "0.7n": (0.6 * num_features, 0.8 * num_features),
                                           "0.9n": (0.8 * num_features, num_features)
                                           }
                        # single sensitive attribute
                        for sens_attr in sensitive_attributes:
                            if fairness_metric == "SPD":
                                idx_privileged = np.nonzero(X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 1)[0]
                                idx_unprivileged = np.nonzero(X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 0)[0]
                                interaction_p_avg = np.mean(interactions[idx_privileged, :], axis=0) if idx_privileged.size != 0 else np.zeros(interactions.shape[1])
                                interaction_up_avg = np.mean(interactions[idx_unprivileged, :], axis=0) if idx_unprivileged.size != 0 else np.zeros(interactions.shape[1])
                                diff_interaction = interaction_p_avg - interaction_up_avg
                            elif fairness_metric == "AOD":
                                idx_privileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 1)
                                                                      & (y_test_sampled[:, 0] == 1))[0]
                                idx_privileged_unfavorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 1)
                                                                      & (y_test_sampled[:, 0] == 0))[0]
                                idx_unprivileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 0)
                                                                      & (y_test_sampled[:, 0] == 1))[0]
                                idx_unprivileged_unfavorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 0)
                                                                      & (y_test_sampled[:, 0] == 0))[0]
                                interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :], axis=0) if idx_privileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                interaction_p_uf_avg = np.mean(interactions[idx_privileged_unfavorable, :], axis=0) if idx_privileged_unfavorable.size != 0 else np.zeros(interactions.shape[1])
                                interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :], axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                interaction_up_uf_avg = np.mean(interactions[idx_unprivileged_unfavorable, :], axis=0) if idx_unprivileged_unfavorable.size != 0 else np.zeros(interactions.shape[1])
                                diff_interaction = 0.5 * (interaction_p_f_avg + interaction_p_uf_avg - interaction_up_f_avg - interaction_up_uf_avg)
                            elif fairness_metric == "EOD":
                                idx_privileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 1)
                                                                      & (y_test_sampled[:, 0] == 1))[0]
                                idx_unprivileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 0)
                                                                        & (y_test_sampled[:, 0] == 1))[0]
                                interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :], axis=0) if idx_privileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :], axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                diff_interaction = interaction_p_f_avg - interaction_up_f_avg
                            else:
                                raise NotImplementedError
                            idx_sa = np.nonzero(masks[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 1)[0]
                            idx_non_sa = np.nonzero(masks[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 0)[0]
                            idx_all = list(range(masks.shape[0]))
                            for interaction_type, idx in [("sa", idx_sa), ("non-sa", idx_non_sa), ("all", idx_all)]:
                                mask_sa = masks[idx, :]
                                diff_interaction_sa = diff_interaction[idx]
                                for middle_point, interval in interval_orders.items():
                                    min_order = interval[0]
                                    max_order = interval[1] if interval[1] != num_features else num_features + 1
                                    orders = np.sum(mask_sa, axis=-1)
                                    idx_interval = np.nonzero((orders >= min_order) & (orders < max_order))[0]
                                    diff_interaction_interval = diff_interaction_sa[idx_interval]
                                    sum_diff_interaction_interval = np.sum(diff_interaction_interval) if idx_interval.size != 0 else 0
                                    results[method_name][interaction_type][middle_point].append(sum_diff_interaction_interval)
                        # multiple sensitive attribute
                        if len(sensitive_attributes) >= 2:
                            sa_combinations = list(itertools.combinations(sensitive_attributes, 2))
                            for comb in sa_combinations:
                                if fairness_metric == "SPD":
                                    idx_privileged = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 1)
                                                                & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 1))[0]
                                    idx_unprivileged = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 0)
                                                                & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 0))[0]
                                    interaction_p_avg = np.mean(interactions[idx_privileged, :], axis=0) if idx_privileged.size != 0 else np.zeros(interactions.shape[1])
                                    interaction_up_avg = np.mean(interactions[idx_unprivileged, :], axis=0) if idx_unprivileged.size != 0 else np.zeros(interactions.shape[1])
                                    diff_interaction = interaction_p_avg - interaction_up_avg
                                elif fairness_metric == "AOD":
                                    idx_privileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 1)
                                                                          & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 1)
                                                                          & (y_test_sampled[:, 0] == 1))[0]
                                    idx_privileged_unfavorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 1)
                                                                            & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 1)
                                                                            & (y_test_sampled[:, 0] == 0))[0]
                                    idx_unprivileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 0)
                                                                            & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 0)
                                                                            & (y_test_sampled[:, 0] == 1))[0]
                                    idx_unprivileged_unfavorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 0)
                                                                              & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 0)
                                                                              & (y_test_sampled[:, 0] == 0))[0]
                                    interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :], axis=0) if idx_privileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                    interaction_p_uf_avg = np.mean(interactions[idx_privileged_unfavorable, :], axis=0) if idx_privileged_unfavorable.size != 0 else np.zeros(interactions.shape[1])
                                    interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :], axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                    interaction_up_uf_avg = np.mean(interactions[idx_unprivileged_unfavorable, :], axis=0) if idx_unprivileged_unfavorable.size != 0 else np.zeros(interactions.shape[1])
                                    diff_interaction = 0.5 * (interaction_p_f_avg + interaction_p_uf_avg - interaction_up_f_avg - interaction_up_uf_avg)
                                elif fairness_metric == "EOD":
                                    idx_privileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 1)
                                                                          & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 1)
                                                                          & (y_test_sampled[:, 0] == 1))[0]
                                    idx_unprivileged_favorable = np.nonzero((X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 0)
                                                                            & (X_test_sampled[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 0)
                                                                            & (y_test_sampled[:, 0] == 1))[0]
                                    interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :], axis=0) if idx_privileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                    interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :], axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(interactions.shape[1])
                                    diff_interaction = interaction_p_f_avg - interaction_up_f_avg
                                idx_sa = np.nonzero((masks[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 1)
                                                    | (masks[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 1))[0]
                                idx_non_sa = np.nonzero((masks[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 0)
                                                        & (masks[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 0))[0]
                                idx_all = list(range(masks.shape[0]))
                                for interaction_type, idx in [("sa", idx_sa), ("non-sa", idx_non_sa), ("all", idx_all)]:
                                    mask_sa = masks[idx, :]
                                    diff_interaction_sa = diff_interaction[idx]
                                    for middle_point, interval in interval_orders.items():
                                        min_order = interval[0]
                                        max_order = interval[1] if interval[1] != num_features else num_features + 1
                                        orders = np.sum(mask_sa, axis=-1)
                                        idx_interval = np.nonzero((orders >= min_order) & (orders < max_order))[0]
                                        diff_interaction_interval = diff_interaction_sa[idx_interval]
                                        sum_diff_interaction_interval = np.sum(diff_interaction_interval) if idx_interval.size != 0 else 0
                                        results[method_name][interaction_type][middle_point].append(sum_diff_interaction_interval)

    with open(osp.join(save_root, fairness_metric+"_order_wise_interaction_sum.json"), "w") as f:
        json.dump(results, f, indent=1)


def compare_via_boxplot(method_names, fairness_metrics):
    """
    Plot Fig.2 in the paper.
    """
    # capital_method_names = ["default", "Fair-Smote", "PR", "MAAT", "REW", "ADV"]
    plt.rcParams.update({'font.size': 22})
    save_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis")
    fig_root = osp.join(save_root, "figs")
    makedirs(fig_root)
    middle_points = ["0.1n", "0.3n", "0.5n", "0.7n", "0.9n"]
    interaction_types = ["sa", "non-sa", "all"]
    y_labels = {"sa": "Interaction difference",
                "non-sa": "Interaction difference",
                "all": "Interaction difference"}
    x = list(range(0, len(method_names)*len(middle_points), len(method_names)))
    width = 0.6
    interval = 0.8
    if len(method_names) <= 7:
        colors = ["#045275", "#089099", "#7CCBA2", "#FCDE9C", "#F0746E", "#DC3977", "#7C1D6F"]
    else:
        raise NotImplementedError("The pre-defined color choices are not enough.")

    for fairness_metric in fairness_metrics:
        with open(osp.join(save_root, fairness_metric+"_order_wise_interaction_sum.json"), "r", encoding="UTF-8") as f:
            results = json.loads(f.read())

        for interaction_type in interaction_types:
            plt.figure(figsize=(16, 6))
            y_plot = {}
            x_ticks = []
            for i, middle_point in enumerate(middle_points):
                y_plot[middle_point] = []
                for method_name in method_names:
                    y_plot[middle_point].append(results[method_name][interaction_type][middle_point])
                x_plot = [x[i]+multi*interval for multi in range(0, len(method_names))]
                bplot = plt.boxplot(y_plot[middle_point], patch_artist=True,
                                    medianprops={'linestyle': '-', 'color': 'r', 'linewidth': 1.5},
                                    labels=method_names, positions=x_plot, widths=width,
                                    showfliers=False)
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_facecolor(color)
                x_ticks.append(0.5*(x_plot[0]+x_plot[-1]))
            plt.xticks(x_ticks, middle_points)
            plt.ylabel(y_labels[interaction_type], fontsize=25)
            plt.grid(axis="y", linestyle="--", alpha=0.3)
            plt.legend(bplot['boxes'], method_names, loc='upper right', ncol=3)
            plt.axhline(y=0, linewidth=2, color='k')
            plt.tight_layout()
            plt.savefig(osp.join(fig_root, fairness_metric+"_"+interaction_type+".png"), dpi=600)
            plt.pause(0.02)
            plt.clf()


def correlation_analysis(method_names, fairness_metrics, explored_orders=["0.1n"]):
    """
    Table II in the paper.
    """
    save_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis")
    fout = open(osp.join(save_root, "interaction_correlation_analysis.txt"), "w")
    interaction_types = ["sa", "non-sa", "all"]
    corr_statistic_methods = [("Spearman correlation coefficients", spearmanr), ("Kendall's tau correlation coefficients", kendalltau)]

    for corr_statistic_method in corr_statistic_methods:
        fout.write("The %s between sensitive interactions and all interactions,"
                   "as well as those between non-sensitive interactions and all interactions,"
                   "for the most impactful orders:\n" % (corr_statistic_method[0]))
        fout.write("\tsensitive_interactions\tnon-sensitive_interactions")
        corr_func = corr_statistic_method[1]
        for fairness_metric in fairness_metrics:
            fout.write("\n" + fairness_metric)
            with open(osp.join(save_root, fairness_metric+"_order_wise_interaction_sum.json"), "r", encoding="UTF-8") as f:
                results = json.loads(f.read())
            interaction_lists = {}
            for interaction_type in interaction_types:
                interaction_lists[interaction_type] = []
                for explored_order in explored_orders:
                    for method_name in method_names:
                        interaction_lists[interaction_type] += results[method_name][interaction_type][explored_order]
            sa_corr = corr_func(interaction_lists["sa"], interaction_lists["all"])
            non_sa_corr = corr_func(interaction_lists["non-sa"], interaction_lists["all"])
            for corr in [sa_corr, non_sa_corr]:
                if corr.pvalue < 0.05:
                    fout.write("\t%.3f*" % (corr.statistic))
                else:
                    fout.write("\t%.3f" % (corr.statistic))
        fout.write("\n\n")

    fout.close()


def volatility_analysis(method_names, fairness_metrics, explored_orders=["0.1n"]):
    save_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis")
    fout = open(osp.join(save_root, "interaction_volatility_analysis.txt"), "w")
    interaction_types = ["sa", "non-sa", "all"]

    fout.write("The standard deviation of all interactions, sensitive interactions "
               "and non-sensitive interactions for the most impactful orders:\n")
    fout.write("\tsensitive_interactions\tnon-sensitive_interactions\tall_interactions")
    for fairness_metric in fairness_metrics:
        fout.write("\n" + fairness_metric)
        with open(osp.join(save_root, fairness_metric + "_order_wise_interaction_sum.json"), "r",
                  encoding="UTF-8") as f:
            results = json.loads(f.read())
        interaction_lists = {}
        for interaction_type in interaction_types:
            interaction_lists[interaction_type] = []
            for explored_order in explored_orders:
                for method_name in method_names:
                    interaction_lists[interaction_type] += results[method_name][interaction_type][explored_order]
            std = statistics.stdev(interaction_lists[interaction_type])
            fout.write("\t%.3f" % (std))
    fout.write("\n\n")

    fout.close()


def purely_sensitive_delta_i_contribution_analysis(method_names, dataset_names, classifier_names, seed_range):
    """
    Table III in the paper.
    """
    save_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis")
    fairness_metrics = ["SPD", "AOD", "EOD"]
    max_order_ratio = 0.2
    fout = open(osp.join(save_root, "purely_sensitive_interaction_contribution_analysis.txt"), "w")

    fout.write(f"For interaction orders below {max_order_ratio}*n,\n"
               "A: absolute value of sum of purely sensitive interactions\n"
               "B: absolute value of sum of other interactions involving sensitive attributes\n"
               "C: absolute value of sum of interactions NOT involving sensitive attributes\n"
               "D: absolute value of sum of all interactions\n"
               "A|B|C|D:\n")
    for fairness_metric in fairness_metrics:
        fout.write("\n\n%s" % (fairness_metric))
        for method_name in method_names:
            list_abs_sum_purely_sensitive = []
            list_abs_sum_remaining_sensitive = []
            list_abs_sum_non_sensitive = []
            list_abs_sum_all = []
            for dataset_name in dataset_names:
                for classifier_name in classifier_names:
                    if method_name in ["meta", "pr", "adv"] and classifier_name in ["svm", "rf", "dl"]:
                        continue
                    else:
                        for seed in seed_range:
                            interaction_root = osp.join(osp.dirname(__file__),
                                                        "../results/harsanyi_interaction_analysis",
                                                        dataset_name, method_name, "seed_" + str(seed),
                                                        classifier_name)
                            masks = np.load(osp.join(interaction_root, "masks.npy"))
                            interactions = np.load(osp.join(interaction_root, "interactions.npy"))
                            interactions = np.squeeze(interactions, axis=-1)
                            X_test_sampled = np.load(osp.join(interaction_root, "X_test_sampled.npy"))
                            y_test_sampled = np.load(osp.join(interaction_root, "y_test_sampled.npy"))
                            sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
                            num_features = X_test_sampled.shape[-1]
                            max_order = max_order_ratio * num_features
                            # single sensitive attribute
                            for sens_attr in sensitive_attributes:
                                if fairness_metric == "SPD":
                                    idx_privileged = np.nonzero(X_test_sampled[:,
                                                                considered_sensitive_attributes[dataset_name][
                                                                    sens_attr]] == 1)[0]
                                    idx_unprivileged = np.nonzero(X_test_sampled[:,
                                                                  considered_sensitive_attributes[dataset_name][
                                                                      sens_attr]] == 0)[0]
                                    interaction_p_avg = np.mean(interactions[idx_privileged, :],
                                                                axis=0) if idx_privileged.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    interaction_up_avg = np.mean(interactions[idx_unprivileged, :],
                                                                 axis=0) if idx_unprivileged.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    diff_interaction = interaction_p_avg - interaction_up_avg
                                elif fairness_metric == "AOD":
                                    idx_privileged_favorable = np.nonzero((X_test_sampled[:,
                                                                           considered_sensitive_attributes[
                                                                               dataset_name][sens_attr]] == 1)
                                                                          & (y_test_sampled[:, 0] == 1))[0]
                                    idx_privileged_unfavorable = np.nonzero((X_test_sampled[:,
                                                                             considered_sensitive_attributes[
                                                                                 dataset_name][sens_attr]] == 1)
                                                                            & (y_test_sampled[:, 0] == 0))[0]
                                    idx_unprivileged_favorable = np.nonzero((X_test_sampled[:,
                                                                             considered_sensitive_attributes[
                                                                                 dataset_name][sens_attr]] == 0)
                                                                            & (y_test_sampled[:, 0] == 1))[0]
                                    idx_unprivileged_unfavorable = np.nonzero((X_test_sampled[:,
                                                                               considered_sensitive_attributes[
                                                                                   dataset_name][sens_attr]] == 0)
                                                                              & (y_test_sampled[:, 0] == 0))[0]
                                    interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :],
                                                                  axis=0) if idx_privileged_favorable.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    interaction_p_uf_avg = np.mean(interactions[idx_privileged_unfavorable, :],
                                                                   axis=0) if idx_privileged_unfavorable.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :],
                                                                   axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    interaction_up_uf_avg = np.mean(interactions[idx_unprivileged_unfavorable, :],
                                                                    axis=0) if idx_unprivileged_unfavorable.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    diff_interaction = 0.5 * (
                                                interaction_p_f_avg + interaction_p_uf_avg - interaction_up_f_avg - interaction_up_uf_avg)
                                elif fairness_metric == "EOD":
                                    idx_privileged_favorable = np.nonzero((X_test_sampled[:,
                                                                           considered_sensitive_attributes[
                                                                               dataset_name][sens_attr]] == 1)
                                                                          & (y_test_sampled[:, 0] == 1))[0]
                                    idx_unprivileged_favorable = np.nonzero((X_test_sampled[:,
                                                                             considered_sensitive_attributes[
                                                                                 dataset_name][sens_attr]] == 0)
                                                                            & (y_test_sampled[:, 0] == 1))[0]
                                    interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :],
                                                                  axis=0) if idx_privileged_favorable.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :],
                                                                   axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(
                                        interactions.shape[1])
                                    diff_interaction = interaction_p_f_avg - interaction_up_f_avg
                                else:
                                    raise NotImplementedError
                                idx_sa = np.nonzero(masks[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 1)[0]
                                idx_non_sa = np.nonzero(masks[:, considered_sensitive_attributes[dataset_name][sens_attr]] == 0)[0]
                                mask_sa = masks[idx_sa, :]
                                mask_non_sa = masks[idx_non_sa, :]
                                diff_interaction_sa = diff_interaction[idx_sa]
                                diff_interaction_non_sa = diff_interaction[idx_non_sa]
                                orders = np.sum(mask_sa, axis=-1)
                                orders_non_sa = np.sum(mask_non_sa, axis=-1)
                                orders_all = np.sum(masks, axis=-1)
                                idx_interval = np.nonzero(orders < max_order)[0]
                                idx_interval_non_sa = np.nonzero(orders_non_sa < max_order)[0]
                                idx_interval_all = np.nonzero(orders_all < max_order)[0]
                                if idx_interval_all.size != 0:
                                    list_abs_sum_all.append(np.abs(np.sum(diff_interaction[idx_interval_all])))
                                if idx_interval_non_sa.size != 0:
                                    list_abs_sum_non_sensitive.append(np.abs(np.sum(diff_interaction_non_sa[idx_interval_non_sa])))
                                if idx_interval.size != 0:
                                    mask_interval = mask_sa[idx_interval, :]
                                    diff_interaction_interval = diff_interaction_sa[idx_interval]
                                    idx_purely_sensitive = np.where(
                                        np.all(mask_interval[:, np.arange(mask_interval.shape[1]) != considered_sensitive_attributes[dataset_name][sens_attr]] == False, axis=1) &
                                        np.any(mask_interval[:, [considered_sensitive_attributes[dataset_name][sens_attr]]] == True, axis=1)
                                    )[0]
                                    idx_remaining_sensitive = np.setdiff1d(np.arange(mask_interval.shape[0]), idx_purely_sensitive)
                                    list_abs_sum_purely_sensitive.append(np.abs(diff_interaction_interval[idx_purely_sensitive]))
                                    list_abs_sum_remaining_sensitive.append(np.abs(np.sum(diff_interaction_interval[idx_remaining_sensitive])))
                            # multiple sensitive attribute
                            if len(sensitive_attributes) >= 2:
                                sa_combinations = list(itertools.combinations(sensitive_attributes, 2))
                                for comb in sa_combinations:
                                    if fairness_metric == "SPD":
                                        idx_privileged = np.nonzero((X_test_sampled[:,
                                                                     considered_sensitive_attributes[dataset_name][
                                                                         comb[0]]] == 1)
                                                                    & (X_test_sampled[:,
                                                                       considered_sensitive_attributes[
                                                                           dataset_name][comb[1]]] == 1))[0]
                                        idx_unprivileged = np.nonzero((X_test_sampled[:,
                                                                       considered_sensitive_attributes[
                                                                           dataset_name][comb[0]]] == 0)
                                                                      & (X_test_sampled[:,
                                                                         considered_sensitive_attributes[
                                                                             dataset_name][comb[1]]] == 0))[0]
                                        interaction_p_avg = np.mean(interactions[idx_privileged, :],
                                                                    axis=0) if idx_privileged.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        interaction_up_avg = np.mean(interactions[idx_unprivileged, :],
                                                                     axis=0) if idx_unprivileged.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        diff_interaction = interaction_p_avg - interaction_up_avg
                                    elif fairness_metric == "AOD":
                                        idx_privileged_favorable = np.nonzero((X_test_sampled[:,
                                                                               considered_sensitive_attributes[
                                                                                   dataset_name][comb[0]]] == 1)
                                                                              & (X_test_sampled[:,
                                                                                 considered_sensitive_attributes[
                                                                                     dataset_name][comb[1]]] == 1)
                                                                              & (y_test_sampled[:, 0] == 1))[0]
                                        idx_privileged_unfavorable = np.nonzero((X_test_sampled[:,
                                                                                 considered_sensitive_attributes[
                                                                                     dataset_name][comb[0]]] == 1)
                                                                                & (X_test_sampled[:,
                                                                                   considered_sensitive_attributes[
                                                                                       dataset_name][comb[1]]] == 1)
                                                                                & (y_test_sampled[:, 0] == 0))[0]
                                        idx_unprivileged_favorable = np.nonzero((X_test_sampled[:,
                                                                                 considered_sensitive_attributes[
                                                                                     dataset_name][comb[0]]] == 0)
                                                                                & (X_test_sampled[:,
                                                                                   considered_sensitive_attributes[
                                                                                       dataset_name][comb[1]]] == 0)
                                                                                & (y_test_sampled[:, 0] == 1))[0]
                                        idx_unprivileged_unfavorable = np.nonzero((X_test_sampled[:,
                                                                                   considered_sensitive_attributes[
                                                                                       dataset_name][comb[0]]] == 0)
                                                                                  & (X_test_sampled[:,
                                                                                     considered_sensitive_attributes[
                                                                                         dataset_name][
                                                                                         comb[1]]] == 0)
                                                                                  & (y_test_sampled[:, 0] == 0))[0]
                                        interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :],
                                                                      axis=0) if idx_privileged_favorable.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        interaction_p_uf_avg = np.mean(interactions[idx_privileged_unfavorable, :],
                                                                       axis=0) if idx_privileged_unfavorable.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :],
                                                                       axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        interaction_up_uf_avg = np.mean(
                                            interactions[idx_unprivileged_unfavorable, :],
                                            axis=0) if idx_unprivileged_unfavorable.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        diff_interaction = 0.5 * (
                                                    interaction_p_f_avg + interaction_p_uf_avg - interaction_up_f_avg - interaction_up_uf_avg)
                                    elif fairness_metric == "EOD":
                                        idx_privileged_favorable = np.nonzero((X_test_sampled[:,
                                                                               considered_sensitive_attributes[
                                                                                   dataset_name][comb[0]]] == 1)
                                                                              & (X_test_sampled[:,
                                                                                 considered_sensitive_attributes[
                                                                                     dataset_name][comb[1]]] == 1)
                                                                              & (y_test_sampled[:, 0] == 1))[0]
                                        idx_unprivileged_favorable = np.nonzero((X_test_sampled[:,
                                                                                 considered_sensitive_attributes[
                                                                                     dataset_name][comb[0]]] == 0)
                                                                                & (X_test_sampled[:,
                                                                                   considered_sensitive_attributes[
                                                                                       dataset_name][comb[1]]] == 0)
                                                                                & (y_test_sampled[:, 0] == 1))[0]
                                        interaction_p_f_avg = np.mean(interactions[idx_privileged_favorable, :],
                                                                      axis=0) if idx_privileged_favorable.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        interaction_up_f_avg = np.mean(interactions[idx_unprivileged_favorable, :],
                                                                       axis=0) if idx_unprivileged_favorable.size != 0 else np.zeros(
                                            interactions.shape[1])
                                        diff_interaction = interaction_p_f_avg - interaction_up_f_avg
                                    idx_sa = np.nonzero(
                                        (masks[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 1)
                                        | (masks[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 1))[
                                        0]
                                    idx_non_sa = np.nonzero(
                                        (masks[:, considered_sensitive_attributes[dataset_name][comb[0]]] == 0)
                                        & (masks[:, considered_sensitive_attributes[dataset_name][comb[1]]] == 0))[
                                        0]
                                    mask_sa = masks[idx_sa, :]
                                    mask_non_sa = masks[idx_non_sa, :]
                                    diff_interaction_sa = diff_interaction[idx_sa]
                                    diff_interaction_non_sa = diff_interaction[idx_non_sa]
                                    orders = np.sum(mask_sa, axis=-1)
                                    orders_non_sa = np.sum(mask_non_sa, axis=-1)
                                    orders_all = np.sum(masks, axis=-1)
                                    idx_interval = np.nonzero(orders < max_order)[0]
                                    idx_interval_non_sa = np.nonzero(orders_non_sa < max_order)[0]
                                    idx_interval_all = np.nonzero(orders_all < max_order)[0]
                                    if idx_interval_all.size != 0:
                                        list_abs_sum_all.append(np.abs(np.sum(diff_interaction[idx_interval_all])))
                                    if idx_interval_non_sa.size != 0:
                                        list_abs_sum_non_sensitive.append(np.abs(np.sum(diff_interaction_non_sa[idx_interval_non_sa])))
                                    if idx_interval.size != 0:
                                        mask_interval = mask_sa[idx_interval, :]
                                        diff_interaction_interval = diff_interaction_sa[idx_interval]
                                        sens_attr_cols = np.array([considered_sensitive_attributes[dataset_name][comb[0]], considered_sensitive_attributes[dataset_name][comb[1]]])
                                        non_sens_attr_cols = np.setdiff1d(np.arange(mask_interval.shape[1]),
                                                                     sens_attr_cols)
                                        idx_purely_sensitive = np.where(
                                            np.all(mask_interval[:, non_sens_attr_cols] == False, axis=1) &
                                            np.any(mask_interval[:, sens_attr_cols] == True, axis=1)
                                        )[0]
                                        idx_remaining_sensitive = np.setdiff1d(np.arange(mask_interval.shape[0]),
                                                                     idx_purely_sensitive)
                                        list_abs_sum_purely_sensitive.append(np.abs(np.sum(diff_interaction_interval[idx_purely_sensitive])))
                                        list_abs_sum_remaining_sensitive.append(np.abs(np.sum(diff_interaction_interval[idx_remaining_sensitive])))

            fout.write("\n%s" % (method_name))
            fout.write("\n\taverage:")
            fout.write("\t%.3f|%.3f|%.3f|%.3f" % (np.mean(np.array(list_abs_sum_purely_sensitive)), np.mean(np.array(list_abs_sum_remaining_sensitive)), np.mean(np.array(list_abs_sum_non_sensitive)), np.mean(np.array(list_abs_sum_all))))
            fout.write("\n\tmedian:")
            fout.write("\t%.3f|%.3f|%.3f|%.3f" % (np.median(np.array(list_abs_sum_purely_sensitive)), np.median(np.array(list_abs_sum_remaining_sensitive)), np.median(np.array(list_abs_sum_non_sensitive)), np.median(np.array(list_abs_sum_all))))

    fout.close()


def purely_sensitive_interaction_analysis(method_names, dataset_names, classifier_names, seed_range):
    """
    Red line in Fig.9 of the paper.
    """
    save_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis")
    fout = open(osp.join(save_root, "avg_abs_purely_sensitive_interaction_values.txt"), "w")
    fout.write("The mean of absolute values of purely sensitive interactions (I, not delta I):\n")
    for method_name in method_names:
        fout.write("%s\t" % (method_name))
    fout.write("\n")
    results = {}
    for method_name in method_names:
        results[method_name] = []
        for dataset_name in dataset_names:
            for classifier_name in classifier_names:
                if method_name in ["meta", "pr", "adv"] and classifier_name in ["svm", "rf", "dl"]:
                    continue
                elif "hifi" in method_name and classifier_name in ["rf"]:
                    continue
                else:
                    for seed in seed_range:
                        if ("hifi" in method_name) or ("hifi" not in method_name and dataset_name not in ["census", "ufrgs", "compas", "diabetes"]):   # datasets whose number of attributes exceeds 12
                            data_root = osp.join(osp.dirname(__file__), "../data/tabular", dataset_name, dataset_name + "_processed.csv")

                            # calculate mean values on training set as the baseline, and sample data from testing set for experiments
                            set_seed(seed)
                            dataset_orig = pd.read_csv(data_root)
                            dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=SPLIT_RATIO[dataset_name], shuffle=True)
                            scaler = MinMaxScaler()
                            scaler.fit(dataset_orig_train)
                            dataset_orig_train = scaler.transform(dataset_orig_train)
                            dataset_orig_test = scaler.transform(dataset_orig_test)
                            mean_baseline = np.mean(dataset_orig_train[:, :-1], axis=0)
                            idx_lst = list(range(dataset_orig_test.shape[0]))
                            random.shuffle(idx_lst)
                            X_test_sampled = dataset_orig_test[idx_lst][:USED_SAMPLE_NUM[dataset_name], :-1]
                            y_test_sampled = dataset_orig_test[idx_lst][:USED_SAMPLE_NUM[dataset_name], [-1]]

                            orig_model = load_original_model(method_name, dataset_name, classifier_name, seed)
                            model, reward_type = model_standardize(method_name, dataset_name, classifier_name, orig_model)

                            indices_sensitive_attributes = list(considered_sensitive_attributes[dataset_name].values())
                            sensitive_coalitions = get_all_nonempty_subsets(indices_sensitive_attributes)
                            for sensitive_coalition in sensitive_coalitions:
                                interactions_purely_sensitive = np.zeros_like(y_test_sampled)
                                subsets = get_all_subsets(sensitive_coalition)
                                for subset in subsets:
                                    masked_inputs = np.copy(X_test_sampled)
                                    mask = np.ones(X_test_sampled.shape[1]).astype(bool)
                                    mask[subset] = False
                                    masked_inputs[:, mask] = mean_baseline[mask]
                                    outputs_on_masked_inputs = model(masked_inputs)
                                    interactions_purely_sensitive = interactions_purely_sensitive + ((-1) ** (len(sensitive_coalition) - len(subset))) * outputs_on_masked_inputs
                                results[method_name].append(interactions_purely_sensitive.reshape(-1))

                            continue

                        interaction_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis",
                                               dataset_name, method_name, "seed_" + str(seed), classifier_name)
                        masks = np.load(osp.join(interaction_root, "masks.npy"))
                        interactions = np.load(osp.join(interaction_root, "interactions.npy"))
                        interactions = np.squeeze(interactions, axis=-1)
                        sens_attr_cols = np.array(list(considered_sensitive_attributes[dataset_name].values()))
                        non_sens_attr_cols = np.setdiff1d(np.arange(masks.shape[1]), sens_attr_cols)
                        idx_purely_sensitive = np.where(
                            np.all(masks[:, non_sens_attr_cols] == False, axis=1) &
                            np.any(masks[:, sens_attr_cols] == True, axis=1)
                        )[0]
                        interactions_purely_sensitive = interactions[:, idx_purely_sensitive].reshape(-1)
                        results[method_name].append(interactions_purely_sensitive)
        combined_array = np.concatenate(results[method_name])
        fout.write("%.3f\t" % (np.mean(np.abs(combined_array))))
        results[method_name] = combined_array.tolist()
    fout.close()
    with open(osp.join(save_root, "raw_purely_sensitive_interaction_values.json"), "w") as f:
        json.dump(results, f, indent=1)


def calculate_avg_I_on_emptyset(dataset_names, classifier_names, seed_range):
    """
    Analyze the validity of estimated baseline values.
    """
    method_name = "default"
    save_root = osp.join(osp.dirname(__file__), "../results/harsanyi_interaction_analysis")
    fout = open(osp.join(save_root, "avg_I_on_emptyset.txt"), "w")
    fout.write("\tE[I(empty)]\tE[|I(empty)|]\n")
    all_list_I = []
    for dataset_name in dataset_names:
        list_I = []
        fout.write(dataset_name)
        for seed in seed_range:
            for classifier_name in classifier_names:
                interaction_root = osp.join(save_root, dataset_name, method_name, "seed_"+str(seed), classifier_name, "interactions.npy")
                I = np.load(interaction_root)
                list_I.append(I[:, 0, :])
        list_I = np.concatenate(list_I)
        all_list_I.append(list_I)
        fout.write("\t%.3f\t%.3f\n" % (np.mean(list_I), np.mean(np.abs(list_I))))
    all_list_I = np.concatenate(all_list_I)
    fout.write("all\t%.3f\t%.3f" % (np.mean(all_list_I), np.mean(np.abs(all_list_I))))
    fout.close()


if __name__ == '__main__':
    method_names = ["default", "fairsmote", "pr", "maat", "rew", "adv"]
    dataset_names = ["census", "ufrgs", "compas", "diabetes"]
    classifier_names = ["dl", "lr", "rf", "svm"]
    seed_range = list(range(0, 10))
    fairness_metrics = ["SPD", "AOD", "EOD"]

    n_process = multiprocessing.cpu_count()      # number of parallel processes
    with multiprocessing.Pool(processes=n_process) as pool:
        for dataset_name in dataset_names:
            for method_name in method_names:
                for classifier_name in classifier_names:
                    if method_name in ["meta", "pr", "adv"] and classifier_name in ["svm", "rf", "dl"]:
                        continue
                    else:
                        for seed in seed_range:
                            pool.apply_async(calculate_interactions_on_sampled_data, args=(method_name, dataset_name, classifier_name, seed))

        pool.close()
        pool.join()

    for fairness_metric in fairness_metrics:
        fairness_gap_interaction_analysis(method_names, dataset_names, classifier_names, seed_range, fairness_metric)

    compare_via_boxplot(method_names, fairness_metrics)

    correlation_analysis(method_names, fairness_metrics)

    volatility_analysis(method_names, fairness_metrics)

    purely_sensitive_delta_i_contribution_analysis(method_names, dataset_names, classifier_names, seed_range)

    calculate_avg_I_on_emptyset(dataset_names, classifier_names, seed_range)

    # red line in Fig. 9
    dataset_names_1 = ["census", "ufrgs", "compas", "diabetes", "default"]
    method_names_1 = ["default", "fairsmote", "pr", "maat", "rew", "adv", "hifi_eta=0.75"]
    purely_sensitive_interaction_analysis(method_names_1, dataset_names_1, classifier_names, seed_range)