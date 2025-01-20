"""
This module calculates the individual discrimination rate, for which any input in the given dataset gets
inconsistent output with any one of the counterfactual inputs that differ only in sensitive attributes
violates individual fairness once.
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import warnings
warnings.filterwarnings('ignore')

import numpy as np

from tools.utils import get_all_nonempty_subsets


def evaluate_causal_fairness(model, test_data, indices_sensitive_attributes):
    X = test_data.features
    y_pred = (model(X) > 0.5)
    sa_coalitions = get_all_nonempty_subsets(indices_sensitive_attributes)
    is_idi = np.zeros_like(y_pred).astype(bool)

    for sa_coalition in sa_coalitions:
        counterfactual_X = np.copy(X)
        counterfactual_X[:, sa_coalition] = 1 - counterfactual_X[:, sa_coalition]
        counterfactual_y = (model(counterfactual_X) > 0.5)
        is_idi |= (y_pred != counterfactual_y)

    return np.sum(is_idi) / X.shape[0]