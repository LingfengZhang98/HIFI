"""
The classical preprocessing method REW is combined with our in-processing method HIFI to further boost effectiveness.
"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))

import copy
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

from tools.utils import set_seed, makedirs, get_all_nonempty_subsets, get_all_subsets
from tools.config import SPLIT_RATIO, considered_sensitive_attributes, BATCH_SIZE, privileged_groups, unprivileged_groups
from tools.get_classifier import get_torch_classifier, HingeLoss
from tools.metrics import measure_final_score


def get_masked_inputs(inputs, list_activated, mean_values):
    masked_inputs = inputs.clone()
    mask = torch.ones(inputs.size(1), dtype=torch.bool)
    mask[list_activated] = False
    masked_inputs[:, mask] = mean_values[mask]
    return masked_inputs


def custom_loss(classifier_name, model, inputs, outputs, labels, indices_sensitive_attributes, eta, weights):
    # compute the original classification loss
    weights = weights.unsqueeze(1)
    if classifier_name in ["lr", "dl"]:
        classification_loss = nn.BCELoss(weight=weights)(outputs, labels)
    elif classifier_name == "svm":
        classification_loss = HingeLoss(weight=weights)(outputs, labels)

    # compute the additional loss terms from HIFI
    hi_loss = 0
    if eta != 0:
        mean_values = torch.mean(inputs, dim=0)
        sensitive_coalitions = get_all_nonempty_subsets(indices_sensitive_attributes)
        for sensitive_coalition in sensitive_coalitions:
            interactions = torch.zeros_like(outputs)
            subsets = get_all_subsets(sensitive_coalition)
            for subset in subsets:
                masked_inputs = get_masked_inputs(inputs, subset, mean_values)
                if classifier_name in ["svm"]:
                    outputs_on_masked_inputs = torch.sigmoid(model(masked_inputs))
                else:
                    outputs_on_masked_inputs = model(masked_inputs)
                interactions = interactions + ((-1)**(len(sensitive_coalition)-len(subset))) * outputs_on_masked_inputs
                interactions = interactions
            hi_loss += torch.mean(torch.abs(interactions))

    return classification_loss + eta * hi_loss


def train_with_rew_hifi(dataset_name, classifier_name, seed_range=[0], eta=0.75):
    method_name = "rew+hifi"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sensitive_attributes = list(considered_sensitive_attributes[dataset_name].keys())
    indices_sensitive_attributes = list(considered_sensitive_attributes[dataset_name].values())
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

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                                label_names=['Probability'],
                                                protected_attribute_names=sensitive_attributes)
        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=sensitive_attributes)

        RW = Reweighing(unprivileged_groups=unprivileged_groups[dataset_name],
                        privileged_groups=privileged_groups[dataset_name])
        RW.fit(dataset_orig_train)
        dataset_transf_train = RW.transform(dataset_orig_train)

        clf = get_torch_classifier(classifier_name, dataset_transf_train.features.shape[1])
        clf.to(device)

        X = torch.from_numpy(dataset_transf_train.features).float()
        y = torch.from_numpy(dataset_transf_train.labels).float()
        sample_weights = torch.from_numpy(dataset_transf_train.instance_weights.ravel()).float()
        batch_size = BATCH_SIZE[dataset_name]
        data_loader = DataLoader(TensorDataset(X, y, sample_weights), batch_size=batch_size, shuffle=True)

        optimizer = optim.NAdam(clf.parameters(), lr=0.002) if classifier_name == "dl" else optim.NAdam(clf.parameters(), lr=1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        num_epochs = 20 if classifier_name == "dl" else 100
        tolerance = 1e-4
        previous_loss = float('inf')
        early_stop = False
        for epoch in range(num_epochs):
            clf.train()
            epoch_loss = 0
            for inputs, labels, weights in data_loader:
                inputs, labels, weights = inputs.to(device), labels.to(device), weights.to(device)
                optimizer.zero_grad()
                outputs = clf(inputs)
                loss = custom_loss(classifier_name, clf, inputs, outputs, labels, indices_sensitive_attributes, eta, weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_epoch_loss = epoch_loss / len(data_loader)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')
            if abs(previous_loss - avg_epoch_loss) < tolerance:
                print(f'Training converged at epoch {epoch + 1}')
                early_stop = True
                break
            previous_loss = avg_epoch_loss
        if not early_stop:
            print('Reached the maximum number of epochs without convergence.')

        clf.eval()
        with torch.no_grad():
            X_test, y_test = torch.from_numpy(dataset_orig_test.features).float().to(device), torch.from_numpy(
                dataset_orig_test.labels).float().to(device)
            if classifier_name in ["lr", "dl"]:
                preds = (clf(X_test) > 0.5).int()
            elif classifier_name == "svm":
                preds = (clf(X_test) > 0).int()

            torch.save(clf.state_dict(), osp.join(save_root, classifier_name + ".pth"))

        test_df_copy = copy.deepcopy(dataset_orig_test)
        test_df_copy.labels = preds.cpu().numpy()

        round_result = measure_final_score(dataset_orig_test, test_df_copy, sensitive_attributes)

        with open(osp.join(save_root, classifier_name+"_metrics.json"), "w") as f:
            json.dump(round_result, f, indent=1)


if __name__ == '__main__':
    datasets = ["census", "ufrgs", "compas", "diabetes", "default"]
    classifiers = ["lr", "dl", "svm"]
    seed_range = list(range(0, 10))

    for dataset in datasets:
        for clf in classifiers:
            for seed in seed_range:
                print(f"{dataset}-{clf}-seed={seed}")
                train_with_rew_hifi(dataset, clf, [seed])