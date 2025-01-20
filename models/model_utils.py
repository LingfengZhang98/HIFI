import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), ".."))
import warnings
warnings.filterwarnings('ignore')
import joblib
import tensorflow as tf
# tf.compat.v1.disable_resource_variables()
# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
import torch
import numpy as np
import pandas as pd
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from aif360.datasets import BinaryLabelDataset
from bias_mitigation_methods.adv import train_with_adv
from tools.config import considered_sensitive_attributes, preprocessed_df_columns, SPLIT_RATIO
from tools.get_classifier import get_torch_classifier


def get_model_output_dl(x, orig_model):
    return orig_model.predict(x).reshape(-1, 1)


def get_model_output_lr_rf(x, orig_model):
    return orig_model.predict_proba(x)[:, [1]]


def get_model_output_svm(x, orig_model):
    return (1 / (1 + np.exp(-orig_model.decision_function(x)))).reshape(-1, 1)


def get_model_output_meta_pr_adv(x, orig_model, dataset_name):
    y = np.zeros((x.shape[0], 1))
    x_bld = pd.DataFrame(np.hstack((x, y)), columns=preprocessed_df_columns[dataset_name])
    x_bld = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=x_bld,
                               label_names=['Probability'],
                               protected_attribute_names=list(considered_sensitive_attributes[dataset_name].keys()))
    pred_dataset = orig_model.predict(x_bld)
    return np.nan_to_num(pred_dataset.scores)


def get_model_output_maat_lr_rf_svm(x, orig_model, clf_suffix):
    pred_probs = []
    for sfx in clf_suffix:
        pred_probs.append(np.array(orig_model[sfx].predict_proba(x)).reshape(-1, 2))
    pred_probs = [x[:, 1].reshape(-1, 1) for x in pred_probs]
    pred_probs = np.hstack(tuple(pred_probs))
    return np.mean(pred_probs, axis=1).reshape(-1, 1)


def get_model_output_maat_dl(x, orig_model, clf_suffix):
    pred_probs = []
    for sfx in clf_suffix:
        pred_probs.append(np.array(orig_model[sfx].predict(x)).reshape(-1, 1))
    pred_probs = np.hstack(tuple(pred_probs))
    return np.mean(pred_probs, axis=1).reshape(-1, 1)


def get_model_output_torch_lr_dl(x, orig_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_model.to(device)
    with torch.no_grad():
        x = torch.from_numpy(x).float().to(device)
        return orig_model(x).cpu().numpy().reshape(-1, 1)


def get_model_output_torch_svm(x, orig_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_model.to(device)
    with torch.no_grad():
        x = torch.from_numpy(x).float().to(device)
        output = torch.sigmoid(orig_model(x))
        return output.cpu().numpy().reshape(-1, 1)


def load_original_model(method_name, dataset_name, classifier_name, seed):
    model_root = osp.join(osp.dirname(__file__), dataset_name, method_name, "seed_"+str(seed))
    if method_name in ["default", "rew", "fairsmote"]:
        if classifier_name in ["lr", "svm", "rf"]:
            model = joblib.load(osp.join(model_root, classifier_name+".pkl"))
        elif classifier_name in ["dl"]:
            model = tf.keras.models.load_model(osp.join(model_root, classifier_name+".keras"))
        else:
            raise NotImplementedError(f"{classifier_name} has not been implemented for {method_name}.")
    elif method_name in ["meta", "pr"]:
        if classifier_name not in ["lr"]:
            raise NotImplementedError(f"{classifier_name} has not been implemented for {method_name}.")
        # Note that the PR model must be loaded with the same Numpy version as when it was saved.
        model = joblib.load(osp.join(model_root, classifier_name + ".pkl"))
    elif method_name == "adv":
        if classifier_name not in ["lr"]:
            raise NotImplementedError(f"{classifier_name} has not been implemented for {method_name}.")
        model = train_with_adv(dataset_name, [seed])[0]
    elif method_name == "maat":
        clf_suffix = list(considered_sensitive_attributes[dataset_name].keys())
        clf_suffix.append("orig")
        model = {}
        if classifier_name in ["lr", "svm", "rf"]:
            for sfx in clf_suffix:
                model[sfx] = joblib.load(osp.join(model_root, classifier_name+"_"+sfx+".pkl"))
        elif classifier_name in ["dl"]:
            for sfx in clf_suffix:
                model[sfx] = tf.keras.models.load_model(osp.join(model_root, classifier_name+"_"+sfx+".keras"))
        else:
            raise NotImplementedError(f"{classifier_name} has not been implemented for {method_name}.")
    elif "hifi" in method_name:
        if classifier_name in ["lr", "svm", "dl"]:
            model = get_torch_classifier(classifier_name, len(preprocessed_df_columns[dataset_name])-1)
            if "eta" in method_name:    # "hifi_eta=??"
                eta_setting = method_name.split("_")[1]
                model_path = osp.join(osp.dirname(__file__), dataset_name, "hifi", "seed_"+str(seed), eta_setting, classifier_name+".pth")
            else:
                model_path = osp.join(model_root, classifier_name + ".pth")
            model.load_state_dict(torch.load(model_path))
        else:
            raise NotImplementedError(f"{classifier_name} has not been implemented for {method_name}.")
    else:
        raise NotImplementedError(f"{method_name} has not been implemented.")

    return model


def model_standardize(method_name, dataset_name, classifier_name, orig_model):
    """
    Return a callable function that take a 2-d numpy.ndarray as input and return a 2-d numpy.ndarray,
    each row of which is probabilities predicted by the original model,
    as well as the fashion of transforming model outputs into rewards.
    """
    if method_name in ["default", "rew", "fairsmote"]:
        if classifier_name in ["dl"]:
            return partial(get_model_output_dl, orig_model=orig_model), "positive_probability"
        elif classifier_name in ["lr", "rf"]:
            return partial(get_model_output_lr_rf, orig_model=orig_model), "positive_probability"
        elif classifier_name in ["svm"]:
            return partial(get_model_output_svm, orig_model=orig_model), "positive_probability"
    elif method_name in ["meta", "pr", "adv"]:
        if classifier_name in ["lr"]:
            return partial(get_model_output_meta_pr_adv, orig_model=orig_model, dataset_name=dataset_name), "positive_probability"
    elif method_name == "maat":
        clf_suffix = list(considered_sensitive_attributes[dataset_name].keys())
        clf_suffix.append("orig")
        if classifier_name in ["lr", "svm", "rf"]:
            return partial(get_model_output_maat_lr_rf_svm, orig_model=orig_model, clf_suffix=clf_suffix), "positive_probability"
        elif classifier_name in ["dl"]:
            return partial(get_model_output_maat_dl, orig_model=orig_model, clf_suffix=clf_suffix), "positive_probability"
    elif "hifi" in method_name:
        if classifier_name in ["lr", "dl"]:
            return partial(get_model_output_torch_lr_dl, orig_model=orig_model), "positive_probability"
        elif classifier_name in ["svm"]:
            return partial(get_model_output_torch_svm, orig_model=orig_model), "positive_probability"
    else:
        raise NotImplementedError(f"{method_name} with {classifier_name} model has not been implemented for {dataset_name}.")


if __name__ == '__main__':
    # test functionalities
    method_names = ["default", "rew", "fairsmote", "maat", "adv", "meta", "pr", "hifi_eta=0.75"]
    dataset_names = ["census", "ufrgs", "compas", "diabetes", "default"]
    classifier_names = ["dl", "lr", "rf", "svm"]
    seed = 0
    for dataset_name in dataset_names:
        for method_name in method_names:
            for classifier_name in classifier_names:
                if method_name in ["meta", "pr", "adv"] and classifier_name in ["svm", "rf", "dl"]:
                    continue
                elif "hifi" in method_name and classifier_name in ["rf"]:
                    continue
                else:
                    print(f"dataset:{dataset_name}\tmethod:{method_name}\tclassifier:{classifier_name}\n")
                    orig_model = load_original_model(method_name, dataset_name, classifier_name, seed)
                    standard_model, _ = model_standardize(method_name, dataset_name, classifier_name, orig_model)
                    dataset_orig = pd.read_csv(osp.join(osp.dirname(__file__), "../data/tabular", dataset_name,
                                                        dataset_name + "_processed.csv"))
                    scaler = MinMaxScaler()
                    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=SPLIT_RATIO[dataset_name], shuffle=True)

                    scaler.fit(dataset_orig_train)
                    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test),
                                                     columns=dataset_orig.columns)
                    pred_probs = standard_model(dataset_orig_test.values[:10, :-1])
                    print(pred_probs)
                    print("====================================================\n")