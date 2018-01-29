import pandas as pd
import numpy as np
from scipy import stats
import importlib
import matplotlib.pyplot as plt
import importlib

class Model(object):
    def __init__(self, heuristic, transformation, score, params, label="Unlabelled", copy=True):
        self.heuristic = heuristic
        self.transformation = transformation
        self.score_func = score
        if copy:
            # TODO Generalize the copy function somehow
            self.params = params.deepcopy()
        else:
            self.params = params
        self.label = label

    def transform(self, X):
        return self.transformation(self.params, X)

    def fit(self, X, Y, hyperparams):
        self.params = self.heuristic(hyperparams, self.score_func)(X, Y)

    def score(self, X, Y):
        return self.score_func(self.transform(X), Y)

    def copy(self):
        # TODO Change the label name.
        to_return = Model(self.heuristic, self.transformation, self.score_func, self.params, self.label)

