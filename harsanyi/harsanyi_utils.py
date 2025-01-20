"""
This code is partially adapted from:
- https://github.com/sjtu-xai-lab/aog/tree/main/src/harsanyi
"""

from typing import Callable, Union, Iterable
import numpy as np
from tqdm import tqdm


def generate_all_masks(n_players):
    """
    Args:
        n_players:  the input dimension (the number of players involved)

    Returns:        a matrix, each line of which corresponds to a kind of masked input, where Ture represents activation and False represents absence
    """
    masks = list(range(2 ** n_players))
    masks = [np.binary_repr(mask, width=n_players) for mask in masks]
    masks = [[bool(int(item)) for item in mask] for mask in masks]
    return np.array(masks)


def generate_first_few_order_masks(n_players, first_few_orders_ratio):
    """
    Args:
        n_players:              the input dimension (the number of players involved)
        first_few_orders_ratio: only the first few orders of interactions considered

    Returns: a matrix, whose lines are contained in [all_masks], and correspond to the interactions of specified first few orders
    """
    all_masks = generate_all_masks(n_players)
    max_order = n_players * first_few_orders_ratio
    if max_order < 1:
        max_order = 1
    all_orders = np.sum(all_masks, axis=1)
    first_few_order_masks = all_masks[all_orders <= max_order]
    return first_few_order_masks


def generate_subset_masks(mask_S, all_masks):
    """
    Args:
        mask_S:     a mask pattern corresponding to a set of players S
        all_masks:  mask matrix

    Returns:        mask patterns of subsets L's of S, as well as their indices in [all_masks]
    """
    assert mask_S.ndim == 1
    assert all_masks.ndim == 2
    expanded_mask_S = np.repeat(mask_S.reshape(1, -1), all_masks.shape[0], axis=0)
    is_subset = np.logical_or(expanded_mask_S, np.logical_not(all_masks))
    is_subset = np.all(is_subset, axis=1)
    return all_masks[is_subset], is_subset


def get_reward2Iand_mat(n_players):
    """
    I(S) = \sum_{L\subseteq S} (-1)^{s-l} * v(L)

    Args:
        n_players:  the input dimension (the number of players involved)

    Returns:        a transformation matrix composed of 0, 1, -1's, with shape 2^n * 2^n, used to compute Harsanyi interactions from rewards by multiplication
    """
    all_masks = generate_all_masks(n_players)
    n_masks = all_masks.shape[0]
    mat = []
    for i in range(n_masks):
        mask_S = all_masks[i]
        row = np.zeros(n_masks)
        mask_Ls, L_indices = generate_subset_masks(mask_S, all_masks)
        row[L_indices] = np.power(-1, np.sum(mask_S) - np.sum(mask_Ls, axis=1))
        mat.append(row)
    return np.array(mat)


def get_first_few_order_reward2Iand_mat(n_players, first_few_orders_ratio):
    """
    I(S) = \sum_{L\subseteq S} (-1)^{s-l} * v(L)

    Args:
        n_players:              the input dimension (the number of players involved)
        first_few_orders_ratio: only the first few orders of interactions considered

    Returns: a transformation matrix composed of 0, 1, -1's, used to compute Harsanyi interactions of the first few orders from rewards by multiplication
    """
    first_few_order_masks = generate_first_few_order_masks(n_players, first_few_orders_ratio)
    n_masks = first_few_order_masks.shape[0]
    mat = []
    for i in range(n_masks):
        mask_S = first_few_order_masks[i]
        row = np.zeros(n_masks)
        mask_Ls, L_indices = generate_subset_masks(mask_S, first_few_order_masks)
        row[L_indices] = np.power(-1, np.sum(mask_S) - np.sum(mask_Ls, axis=1))
        mat.append(row)
    return np.array(mat)


def mask_input_fn_tabular(input, baseline, S_list):
    """
    This function specializes how to mask each input variable.
    When computing $v(S)$, we mask input variables in $N\setminus S$,
    while retaining input variables in $S$.

    Args:
        input:      the original input
        baseline:   baseline values
        S_list:     list of sets of player indices that will not be masked

    Returns:        a matrix of masked inputs, each line of which corresponds to each line of [S_list]
    """
    assert input.shape[0] == baseline.shape[0] == 1
    batch_size = len(S_list)
    input_batch = np.repeat(input, batch_size, axis=0)
    baseline_batch = np.repeat(baseline, batch_size, axis=0)
    mask = np.zeros_like(input_batch)
    for i, S in enumerate(S_list):
        mask[i, S] = 1
    return mask * input_batch + (1 - mask) * baseline_batch


def flatten_(x):
    """
    Flatten an irregular list of lists

    Reference <https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists>

    [In]  flatten_(((1, 2), 3, 4)) -- Note: (with many brackets) x = ( (1, 2) , 3 , 4 )
    [Out] (1, 2, 3, 4)
    """
    if isinstance(x, Iterable):
        return list([a for i in x for a in flatten_(i)])
    else:
        return [x]


def calculate_given_subset_outputs(
        model: Callable,
        input: np.ndarray,
        baseline: np.ndarray,
        player_masks: np.ndarray,
        all_players: Union[None, tuple, list] = None,
        background: Union[None, tuple, list] = None,
        mask_input_fn: Callable = None,
        calc_bs: Union[None, int] = None
) -> (np.ndarray, np.ndarray):
    """
    Args:
        model:          a model to be interpreted
        input:          an input to be interpreted
        baseline:       the baseline values representing absence states
        player_masks:   mask patterns corresponding to sets of players S
        all_players:    the players involved in the game
        background:     the players that always exist
        mask_input_fn:  the specification about how to mask each input variable
        calc_bs:        the batch size of model output calculation

    Returns: [player_masks], model outputs on the masked inputs represented by [player_masks]
    """
    assert input.shape[0] == baseline.shape[0] == 1
    if all_players is None:
        assert (background is None or len(background) == 0) and mask_input_fn is None
        masks = player_masks
    else:
        if background is None:
            background = []
        assert mask_input_fn is not None
        all_players = np.array(all_players, dtype=object)
        grid_indices_list = []
        for i in range(player_masks.shape[0]):
            player_mask = player_masks[i]
            grid_indices_list.append(list(flatten_([background, all_players[player_mask]])))
    if calc_bs is None:
        calc_bs = player_masks.shape[0]
    outputs = []
    # pbar = tqdm(range(int(np.ceil(player_masks.shape[0] / calc_bs))), ncols=100, desc="Calculate model outputs")
    pbar = range(int(np.ceil(player_masks.shape[0] / calc_bs)))
    for batch_id in pbar:
        if all_players is None:
            masks_batch = masks[batch_id*calc_bs:(batch_id+1)*calc_bs]
            masked_inputs_batch = np.where(masks_batch, np.repeat(input, calc_bs, axis=0), np.repeat(baseline, calc_bs, axis=0))
        else:
            grid_indices_batch = grid_indices_list[batch_id * calc_bs:(batch_id + 1) * calc_bs]
            masked_inputs_batch = mask_input_fn(input, baseline, grid_indices_batch)
        output = model(masked_inputs_batch)
        outputs.append(output)
    outputs = np.concatenate(outputs, axis=0)
    return player_masks, outputs


def calculate_all_subset_outputs(
        model: Callable,
        input: np.ndarray,
        baseline: np.ndarray,
        all_players: Union[None, tuple, list] = None,
        background: Union[None, tuple, list] = None,
        mask_input_fn: Callable = None,
        calc_bs: Union[None, int] = None
) -> (np.ndarray, np.ndarray):
    """
    Args:
        model:          a model to be interpreted
        input:          an input to be interpreted
        baseline:       the baseline values representing absence states
        all_players:    the players involved in the game
        background:     the players that always exist
        mask_input_fn:  the specification about how to mask each input variable
        calc_bs:        the batch size of model output calculation

    Returns: masks and model outputs on all the masked inputs
    """
    assert input.shape[0] == baseline.shape[0] == 1
    n_players = input.shape[1] if all_players is None else len(all_players)
    player_masks = generate_all_masks(n_players)
    return calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        player_masks=player_masks, all_players=all_players,
        background=background, mask_input_fn=mask_input_fn,
        calc_bs=calc_bs
    )


def calculate_first_few_order_outputs(
        model: Callable,
        input: np.ndarray,
        baseline: np.ndarray,
        first_few_orders_ratio: float,
        all_players: Union[None, tuple, list] = None,
        background: Union[None, tuple, list] = None,
        mask_input_fn: Callable = None,
        calc_bs: Union[None, int] = None
) -> (np.ndarray, np.ndarray):
    """
    Args:
        model:                  a model to be interpreted
        input:                  an input to be interpreted
        baseline:               the baseline values representing absence states
        first_few_orders_ratio: only the first few orders of interactions considered
        all_players:            the players involved in the game
        background:             the players that always exist
        mask_input_fn:          the specification about how to mask each input variable
        calc_bs:                the batch size of model output calculation

    Returns: masks and model outputs on the masked inputs corresponding to coalitions of specified first few orders
    """
    assert input.shape[0] == baseline.shape[0] == 1
    n_players = input.shape[1] if all_players is None else len(all_players)
    player_masks = generate_first_few_order_masks(n_players, first_few_orders_ratio)
    return calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        player_masks=player_masks, all_players=all_players,
        background=background, mask_input_fn=mask_input_fn,
        calc_bs=calc_bs
    )


def calculate_output_N(
        model: Callable,
        input: np.ndarray,
        baseline: np.ndarray,
        all_players: Union[None, tuple, list] = None,
        background: Union[None, tuple, list] = None,
        mask_input_fn: Callable = None
) -> (np.ndarray, np.ndarray):
    """
    Args:
        model:          a model to be interpreted
        input:          an input to be interpreted
        baseline:       the baseline values representing absence states
        all_players:    the players involved in the game
        background:     the players that always exist
        mask_input_fn:  the specification about how to mask each input variable

    Returns: the model output on the original input
    """
    assert input.shape[0] == baseline.shape[0] == 1
    n_players = input.shape[1] if all_players is None else len(all_players)
    player_masks = np.ones((1, n_players)).astype(bool)
    _, output_N = calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None,
        player_masks=player_masks
    )
    return output_N


def calculate_output_empty(
        model: Callable,
        input: np.ndarray,
        baseline: np.ndarray,
        all_players: Union[None, tuple, list] = None,
        background: Union[None, tuple, list] = None,
        mask_input_fn: Callable = None
) -> (np.ndarray, np.ndarray):
    """
    Args:
        model:          a model to be interpreted
        input:          an input to be interpreted
        baseline:       the baseline values representing absence states
        all_players:    the players involved in the game
        background:     the players that always exist
        mask_input_fn:  the specification about how to mask each input variable

    Returns: the model output on the input with all players absent
    """
    assert input.shape[0] == baseline.shape[0] == 1
    n_players = input.shape[1] if all_players is None else len(all_players)
    player_masks = np.zeros((1, n_players)).astype(bool)
    _, output_empty = calculate_given_subset_outputs(
        model=model, input=input, baseline=baseline,
        all_players=all_players, background=background,
        mask_input_fn=mask_input_fn, calc_bs=None,
        player_masks=player_masks
    )
    return output_empty


def get_reward(outputs, reward_type, **kwargs):
    """
    Scale the outputs into [0,1], where [0,0.5) corresponds to the unfavorable label,
    and [0.5,1] corresponds to the favorable label. The further away from 0.5,
    the more confident the prediction is.
    Args:
        outputs:        model outputs
        reward_type:    the specification of model outputs and reward calculation fashion
        **kwargs:       v0: the reward to be calibrated on all outputs
                        gt: the ground truth label of the original input
    Returns:            scaled outputs
    """
    if reward_type == "positive_probability":
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1
    elif reward_type == "positive_probability-v0":
        assert len(outputs.shape) == 2 and outputs.shape[1] == 1
        assert "v0" in kwargs
        outputs = outputs - kwargs["v0"]
    else:
        raise Exception(f"The reward type '{reward_type}' has not been implemented.")

    return outputs