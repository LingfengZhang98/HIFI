"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Dataset/dataproc.py
- https://github.com/sjtu-xai-lab/Learn/blob/main/util/preprocess_census.py
- https://github.com/LingfengZhang98/EIDIG/blob/main/preprocessing/pre_bank_marketing.py
- https://www.kaggle.com/code/desalegngeb/heart-disease-predictions/notebook

Preprocess the original data, and save it to "{DATASET}_processed.csv".
"""

import os.path as osp

import pandas as pd
import numpy as np


def preprocess_census():
    DATASET = "census"
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        osp.join("tabular", DATASET, "raw_dataset", "adult.data"),
        names=[d[0] for d in dtypes],
        na_values=["?"],
        skipinitialspace=True,
        dtype=dict(dtypes)
    )
    raw_data = raw_data.dropna().reset_index(drop=True)  # drop data points that contains any N/A value
    dataset_orig = raw_data.drop(["Education", "fnlwgt"], axis=1)  # drop redundant or unrelated attributes
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Education"]), dtypes))
    dataset_orig["Age"] = np.where(dataset_orig["Age"] >= 40, 1, 0)
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                dataset_orig[k] = np.array([rcode[v.strip()] for v in dataset_orig[k]])
            elif k == "Race":
                dataset_orig[k] = np.where(dataset_orig[k] != "White", 0, 1)
            elif k == "Sex":
                dataset_orig[k] = np.where(dataset_orig[k] == "Male", 1, 0)
            elif k == "Target":
                dataset_orig[k] = np.where(dataset_orig[k] == "<=50K", 0, 1)
            else:
                dataset_orig[k] = dataset_orig[k].cat.codes
    dataset_orig.rename(index=str, columns={"Target": "Probability"}, inplace=True)

    dataset_orig.to_csv(osp.join("tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_ufrgs():
    DATASET = "ufrgs"
    columns_orig = ["gender", "race", "physics", "biology", "history", "second_language",
                    "geography", "literature", "Portuguese_essay", "math",
                    "chemistry", "mean_GPA"]
    dataset_orig = pd.read_csv(osp.join("tabular", DATASET, "raw_dataset", "data_with_race.csv"), header=None, names=columns_orig)
    dataset_orig = dataset_orig.dropna()
    dataset_orig['race'] = np.where(dataset_orig['race'] != 'White', 0, 1)
    dataset_orig.rename(index=str, columns={"mean_GPA": "Probability"}, inplace=True)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] < 3, 0, 1)

    dataset_orig.to_csv(osp.join("tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_compas():
    DATASET = "compas"
    dataset_orig = pd.read_csv(osp.join("tabular", DATASET, "raw_dataset", "compas-scores-two-years.csv"))
    dataset_orig = dataset_orig.drop(
        ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob', 'age', 'juv_fel_count', 'decile_score',
         'juv_misd_count', 'juv_other_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out', 'c_case_number',
         'c_offense_date', 'c_arrest_date', 'c_days_from_compas', 'c_charge_desc', 'is_recid', 'r_case_number',
         'r_charge_degree', 'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
         'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc',
         'type_of_assessment', 'decile_score', 'score_text', 'screening_date', 'v_type_of_assessment', 'v_decile_score',
         'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'start', 'end', 'event'], axis=1)
    dataset_orig = dataset_orig.dropna()
    dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
    dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1) & (dataset_orig['priors_count'] <= 3),
                                            3, dataset_orig['priors_count'])
    dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
    dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 0, 1)
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45', 45, dataset_orig['age_cat'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
    dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
    dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)
    dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

    dataset_orig.to_csv(osp.join("tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_diabetes():
    DATASET = "diabetes"
    dataset_orig = pd.read_csv(osp.join("tabular", DATASET, "raw_dataset", "diabetes.csv"))

    dataset_orig["Age"] = np.where(dataset_orig["Age"] >= 40, 0, 1)  # mean age is approximately 33.2 years old
    dataset_orig.rename(index=str, columns={"Outcome": "Probability"}, inplace=True)
    dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

    dataset_orig.to_csv(osp.join("tabular", DATASET, DATASET+"_processed.csv"), index=False)


def preprocess_default():
    DATASET = "default"
    dataset_orig = pd.read_csv(osp.join("tabular", DATASET, "raw_dataset", "default_of_credit_card_clients.csv"))

    dataset_orig = dataset_orig.dropna()
    dataset_orig['SEX'] = np.where(dataset_orig['SEX'] == 2, 0, 1)
    dataset_orig['AGE'] = np.where(dataset_orig['AGE'] >= 25, 0, 1)
    dataset_orig = dataset_orig.drop(['ID'], axis=1)
    dataset_orig = dataset_orig.rename(columns={"SEX": "sex"})
    dataset_orig = dataset_orig.rename(columns={"AGE": "age"})
    dataset_orig.rename(index=str, columns={"default payment next month": "Probability"}, inplace=True)

    dataset_orig.to_csv(osp.join("tabular", DATASET, DATASET + "_processed.csv"), index=False)


def majority_label_statistics(dataset_names):
    fout = open(osp.join(osp.dirname(__file__), "tabular_majority_label.txt"), "w")

    for dataset_name in dataset_names:
        fout.write(f"{dataset_name}\t")
    fout.write("\n")
    for dataset_name in dataset_names:
        dataset_orig = pd.read_csv(osp.join(osp.dirname(__file__), "tabular", dataset_name, dataset_name + "_processed.csv"))
        label_avg = dataset_orig["Probability"].mean()
        if label_avg >= 0.5:
            fout.write("1(%.1f" % (100 * label_avg))
            fout.write("\\%)\t")
        else:
            fout.write("0(%.1f" % (100 * (1 - label_avg)))
            fout.write("\\%)\t")


if __name__ == '__main__':
    preprocess_census()
    preprocess_ufrgs()
    preprocess_compas()
    preprocess_diabetes()
    preprocess_default()

    dataset_names = ["census", "ufrgs", "compas", "diabetes", "default"]
    majority_label_statistics(dataset_names)