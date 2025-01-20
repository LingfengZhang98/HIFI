# train-test split ratio
SPLIT_RATIO = {
    "census": 0.3,
    "compas": 0.3,
    "ufrgs": 0.3,
    "diabetes": 0.3,
    "default": 0.3
}

# batch size for training
BATCH_SIZE = {
    "census": 256,
    "compas": 128,
    "ufrgs": 256,
    "diabetes": 32,
    "default": 64
}

# sampling number
USED_SAMPLE_NUM = {
    "census": 500,
    "compas": 500,
    "ufrgs": 500,
    "diabetes": 100,
    "default": 500
}

# name and index of sensitive attributes
considered_sensitive_attributes = {
    "census": {"Race": 6, "Sex": 7},
    "compas": {"sex": 0, "race": 2},
    "ufrgs": {"gender": 0, "race": 1},
    "diabetes": {"Age": 7},
    "default": {"sex": 1, "age": 4}
}

# privileged groups
privileged_groups = {
    "census": [{"Race": 1}, {"Sex": 1}],    # white/male
    "compas": [{"sex": 1}, {"race": 1}],    # Male/Caucasian
    "ufrgs": [{"gender": 1}, {"race": 1}],  # Male/White
    "diabetes": [{"Age": 1}],               # youth
    "default": [{"sex": 1}, {"age": 1}]     # Male/Youth
}

# unprivileged groups
unprivileged_groups = {
    "census": [{"Race": 0}, {"Sex": 0}],    # non-white/female&non-binary
    "compas": [{"sex": 0}, {"race": 0}],    # Female/non-Caucasian
    "ufrgs": [{"gender": 0}, {"race": 0}],  # Female/non-White
    "diabetes": [{"Age": 0}],               # senior
    "default": [{"sex": 0}, {"age": 0}]     # Female/Senior
}

# majority label
majority_label = {
    "census": 0,
    "compas": 1,
    "ufrgs": 0,
    "diabetes": 1,
    "default": 0
}

# columns of preprocessed datasets
preprocessed_df_columns = {
    "census": [
        "Age", "Workclass", "Education-Num", "Marital Status", "Occupation", "Relationship", "Race",
        "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Probability"
        ],
    "compas": [
            "sex", "age_cat", "race", "priors_count", "c_charge_degree", "decile_score.1", "priors_count.1", "Probability"
        ],
    "ufrgs": [
            "gender", "race", "physics", "biology", "history", "second_language",
            "geography", "literature", "Portuguese_essay", "math",
            "chemistry", "Probability"
        ],
    "diabetes": [
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI",
        "DiabetesPedigreeFunction","Age","Probability"
    ],
    "default": [
        "LIMIT_BAL","sex","EDUCATION","MARRIAGE","age","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
        "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","PAY_AMT1","PAY_AMT2",
        "PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","Probability"
    ]
}