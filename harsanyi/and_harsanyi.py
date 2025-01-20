"""
This code is partially adapted from:
- https://github.com/sjtu-xai-lab/aog/tree/main/src/harsanyi
"""

import os
import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import warnings
warnings.filterwarnings('ignore')

from typing import Callable, Union

import numpy as np

from harsanyi.harsanyi_utils import get_reward2Iand_mat, get_first_few_order_reward2Iand_mat, calculate_output_empty, calculate_output_N, get_reward, calculate_all_subset_outputs, calculate_first_few_order_outputs

class AndHarsanyi(object):
    def __init__(
            self,
            model: Callable,
            reward_type: Union[None, str],
            x: np.ndarray,
            y: Union[np.ndarray, int, None],
            baseline: np.ndarray,
            all_players: Union[None, tuple, list] = None,
            background: Union[None, tuple, list] = None,
            mask_input_fn: Callable = None,
            calc_bs: Union[None, int] = None,
            verbose: int = 0,
            first_few_orders_ratio: float = 0.2
    ):
        """
        Args:
            model:          a model to be interpreted
            reward_type:    the way to calculate rewards
            x:              an input to be interpreted
            y:              the ground truth label
            baseline:       the baseline values representing absence states
            all_players:    the players involved in the game
            background:     the players that always exist
            mask_input_fn:  the specification about how to mask each input variable
            calc_bs:        the batch size of model output calculation
            verbose:        version control
            first_few_orders_ratio:     only compute the first few orders of interactions if verbose==2
        """
        assert x.shape[0] == baseline.shape[0] == 1

        self.model = model
        self.reward_type = reward_type
        self.input = x
        self.target = y
        self.baseline = baseline
        self.all_players = all_players
        self.background = [] if background is None else background
        self.mask_input_fn = mask_input_fn
        self.calc_bs = calc_bs
        self.verbose = verbose
        self.n_players = self.input.shape[1] if all_players is None else len(all_players)
        if self.verbose != 2:
            self.reward2Iand = get_reward2Iand_mat(self.n_players)
        else:
            self.first_few_orders_ratio = first_few_orders_ratio
            self.reward2Iand = get_first_few_order_reward2Iand_mat(self.n_players, self.first_few_orders_ratio)

        if self.verbose == 0:
            self.output_empty = calculate_output_empty(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn
            )
            self.output_N = calculate_output_N(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn
            )
            if self.reward_type.endswith("-v0"):
                self.v0 = get_reward(self.output_empty, self.reward_type[:-3], gt=self.target)
            else:
                self.v0 = 0
            self.v_N = get_reward(self.output_N, self.reward_type, gt=self.target, v0=self.v0)
            self.v_empty = get_reward(self.output_empty, self.reward_type, gt=self.target, v0=self.v0)

    def attribute(self):
        if self.verbose != 2:
            self.masks, outputs = calculate_all_subset_outputs(
                model=self.model, input=self.input, baseline=self.baseline,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs
            )
            if self.verbose == 0:
                self.rewards = get_reward(outputs, self.reward_type, gt=self.target, v0=self.v0)
            elif self.verbose == 1:
                self.rewards = get_reward(outputs, self.reward_type, gt=self.target)
            else:
                raise NotImplementedError(f"Unknown verbose=={self.verbose}")
        else:
            self.masks, outputs = calculate_first_few_order_outputs(
                model=self.model, input=self.input, baseline=self.baseline,
                first_few_orders_ratio = self.first_few_orders_ratio,
                all_players=self.all_players, background=self.background,
                mask_input_fn=self.mask_input_fn, calc_bs=self.calc_bs
            )
            self.rewards = get_reward(outputs, self.reward_type)
        self.Iand = np.matmul(self.reward2Iand, self.rewards)

    def get_interaction(self):
        return self.Iand

    def get_masks(self):
        return self.masks

    def get_rewards(self):
        return self.rewards