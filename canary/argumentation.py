from enum import Enum
from bratreader.repomodel import RepoModel
from sklearn.ensemble import RandomForestClassifier


class ClassifierMethod(Enum):
    SVM = "support_vector_machine"
    LR = "logistic_regression"
    NB = "naive_bayes"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if 'value' in other:
            return self.value == other.value
        else:
            return False


