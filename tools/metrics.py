"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Fair3602/Measure_new.py
"""

import numpy as np
import itertools
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef


def cal_spd(dataset_test_pred, p_attr):
    labelname = 'Probability'
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]

    if len(dataset_test_pred[(dataset_test_pred[p_attr] == 0)]) == 0:
        num1 = 0
    else:
        num1 = len(dataset_test_pred[(dataset_test_pred[p_attr] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
            dataset_test_pred[(dataset_test_pred[p_attr] == 0)])

    if len(dataset_test_pred[(dataset_test_pred[p_attr] == 1)]) == 0:
        num2 = 0
    else:
        num2 = len(dataset_test_pred[(dataset_test_pred[p_attr] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
            dataset_test_pred[(dataset_test_pred[p_attr] == 1)])

    return [num1, num2, max([num1, num2])-min([num1, num2])]


def cal_eod(dataset_test, dataset_test_pred, p_attr):
    labelname = 'Probability'
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]

    if len(dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 1)]) == 0:
        num1 = 0
    else:
        num1 = len(
            dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(
            dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 1)])

    if len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1)]) == 0:
        num2 = 0
    else:
        num2 = len(
            dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1)])

    return [num1, num2, max([num1, num2]) - min([num1, num2])]


def cal_aod(dataset_test, dataset_test_pred, p_attr):
    labelname = 'Probability'
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]

    if len(dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 0)]) == 0:
        num1_00 = 0
    else:
        num1_00 = len(
            dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 0)])

    if len(dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 1)]) == 0:
        num1_01 = 0
    else:
        num1_01 = len(
            dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[labelname] == 1)])

    num1 = num1_00 + num1_01

    if len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 0)]) == 0:
        num2_10 = 0
    else:
        num2_10 = len(
            dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 0)])

    if len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1)]) == 0:
        num2_11 = 0
    else:
        num2_11 = len(
            dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1)])

    num2 = num2_10 + num2_11

    return [num1, num2, (max([num1, num2]) - min([num1, num2]))/2]


def wc_spd(dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = 1
    labelname = 'Probability'
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test_pred[labelname] = np.where(dataset_test_pred[labelname] == favorlabel, 1, 0)

    if len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0)]) == 0:
        num1 = 0
    else:
        num1 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
            dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0)])

    if len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1)]) == 0:
        num2 = 0
    else:
        num2 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
            dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1)])

    if len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0)]) == 0:
        num3 = 0
    else:
        num3 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
            dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0)])

    if len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1)]) == 0:
        num4 = 0
    else:
        num4 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
            dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1)])

    return [num1, num2, num3, num4, max([num1, num2, num3, num4]) - min([num1, num2, num3, num4])]


def wc_aod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = dataset_test.favorable_label
    labelname = dataset_test.label_names[0]
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred'+labelname] = np.where(dataset_test['pred'+labelname] == favorlabel, 1, 0)
    num_list = []

    if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0)]) == 0:
        num1_000 = 0
    else:
        num1_000 = len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0)])

    if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) == 0:
        num1_001 = 0
    else:
        num1_001 = len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)])

    num1 = num1_000 + num1_001
    num_list.append(num1)

    if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0)]) == 0:
        num2_010 = 0
    else:
        num2_010 = len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0)])

    if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) == 0:
        num2_011 = 0
    else:
        num2_011 = len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)])

    num2 = num2_010 + num2_011
    num_list.append(num2)

    if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0)]) == 0:
        num3_100 = 0
    else:
        num3_100 = len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 0)])

    if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) == 0:
        num3_101 = 0
    else:
        num3_101 = len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)])

    num3 = num3_100 + num3_101
    num_list.append(num3)

    if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0)]) == 0:
        num4_110 = 0
    else:
        num4_110 = len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 0)])

    if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) == 0:
        num4_111 = 0
    else:
        num4_111 = len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)])

    num4 = num4_110 + num4_111
    num_list.append(num4)

    return [num1, num2, num3, num4, (max(num_list) - min(num_list))/2]


def wc_eod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    favorlabel = 1
    labelname = 'Probability'
    dataset_test = dataset_test.convert_to_dataframe()[0]
    dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred' + labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred' + labelname] = np.where(dataset_test['pred' + labelname] == favorlabel, 1, 0)
    num_list=[]

    if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) == 0:
        num1 = 0
    else:
        num1 = len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)])
    num_list.append(num1)

    if len(dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) == 0:
        num2 = 0
    else:
        num2 = len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)])
    num_list.append(num2)

    if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)]) == 0:
        num3 = 0
    else:
        num3 = len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test[labelname] == 1)])
    num_list.append(num3)

    if len(dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)]) == 0:
        num4 = 0
    else:
        num4 = len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
            dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test[labelname] == 1)])
    num_list.append(num4)

    return [num1, num2, num3, num4, max(num_list) - min(num_list)]


def measure_final_score(dataset_orig_test, dataset_orig_predict, p_attrs):
    """
    Compute common metrics of model utilities and fairness for binary classification.
    """
    y_test = dataset_orig_test.labels
    y_pred = dataset_orig_predict.labels

    # common metrics of model utilities
    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    model_metrics = {"accuracy": accuracy, "recall": recall_macro, "precision": precision_macro,
                     "f1_score": f1score_macro, "Matthews correlation coefficient": mcc}

    # common fairness metrics for single sensitive attributes
    for attr in p_attrs:
        single_spd = cal_spd(dataset_orig_predict, attr)
        single_aod = cal_aod(dataset_orig_test, dataset_orig_predict, attr)
        single_eod = cal_eod(dataset_orig_test, dataset_orig_predict, attr)
        model_metrics[attr + "_SPD"] = single_spd
        model_metrics[attr + "_AOD"] = single_aod
        model_metrics[attr + "_EOD"] = single_eod

    # common metrics for intersectional fairness (w.r.t. subgroups with two sensitive attributes specified)
    if len(p_attrs) >= 2:
        attr_combinations = list(itertools.combinations(p_attrs, 2))
        for comb in attr_combinations:
            intersectional_spd = wc_spd(dataset_orig_predict, comb)
            intersectional_aod = wc_aod(dataset_orig_test, dataset_orig_predict, comb)
            intersectional_eod = wc_eod(dataset_orig_test, dataset_orig_predict, comb)
            model_metrics[comb[0] + "_" + comb[1] + "_SPD"] = intersectional_spd
            model_metrics[comb[0] + "_" + comb[1] + "_AOD"] = intersectional_aod
            model_metrics[comb[0] + "_" + comb[1] + "_EOD"] = intersectional_eod

    return model_metrics