"""
This code is partially adapted from:
- https://github.com/chenzhenpeng18/ICSE24-Multi-Attribute-Fairness/blob/main/Fair3602/utility.py

Already included ML models: LR, SVM, RF, and NN.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from tensorflow import keras
import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self, weight=None):
        super(HingeLoss, self).__init__()
        self.weight = weight

    def forward(self, outputs, labels):
        labels = labels * 2 - 1
        loss = torch.clamp(1 - outputs * labels, min=0)

        if self.weight is not None:
            loss = loss * self.weight

        return torch.mean(loss)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

        # Weight initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Weight initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.zeros_(self.fc4.bias)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.zeros_(self.fc5.bias)
        nn.init.xavier_uniform_(self.fc6.weight)
        nn.init.zeros_(self.fc6.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        return x


def get_classifier(name, feature_size):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    elif name == "dl":
        clf = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=feature_size),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
        clf.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])
    else:
        raise NotImplementedError(f"{name} has not been implemented.")
    return clf


def get_torch_classifier(name, feature_size):
    if name == "lr":
        clf = LogisticRegressionModel(feature_size)
    elif name == "svm":
        clf = SVM(feature_size)
    elif name == "dl":
        clf = SimpleNN(feature_size)
    else:
        raise NotImplementedError(f"{name} has not been implemented with pytorch.")
    return clf