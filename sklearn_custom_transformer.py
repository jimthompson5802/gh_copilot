# sklearn custom transformer for demonstration

from sklearn.base import BaseEstimator, TransformerMixin

# Commentary:  modify to perform required function
class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {'param1': self.param1, 'param2': self.param2}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self):
        return "CustomTransformer(param1={}, param2={})".format(self.param1, self.param2)

    def __str__(self):
        return "CustomTransformer(param1={}, param2={})".format(self.param1, self.param2)

    def __eq__(self, other):
        return self.param1 == other.param1 and self.param2 == other.param2

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.param1, self.param2))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state





