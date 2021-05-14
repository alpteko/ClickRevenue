import numpy as np
import pandas as pd
from scipy.special import inv_boxcox
from scipy.stats import boxcox, kurtosis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class ConTransform:
    def __init__(self):
        self.parameter_dict = {}
        self.eps = 1e-4
        self.flag = True

    def fit(self, series):
        self.name = series.name
        kurt = kurtosis(series)
        data, lamb = boxcox(series.add(self.eps))
        new_kurt = kurtosis(data)
        # print(self.name, kurt, new_kurt)
        if (kurt < 0) & (new_kurt < kurt):
            self.flag = False
        if (kurt > 0) & (new_kurt > kurt):
            self.flag = False
        if self.flag:
            u = data.mean()
            std = data.std() + self.eps
            self.parameter_dict = {"mean": u,
                                   "std": std,
                                   "lambda": lamb
                                   }
        else:
            u = series.mean()
            std = series.std() + self.eps
            self.parameter_dict = {"mean": u,
                                   "std": std
                                   }

    def transform(self, series):
        data = series.copy()
        if self.flag:
            data = boxcox(
                data.add(self.eps), lmbda=self.parameter_dict["lambda"])
            data = pd.Series(data, name=self.name)
        data = data.sub(
            self.parameter_dict["mean"]).div(self.parameter_dict["std"])
        return data.clip(lower=-3.0, upper=3.0).reset_index(drop=True)

    def inv_transform(self, series):
        data = series.copy()
        u = self.parameter_dict["mean"]
        std = self.parameter_dict["std"]
        data = data*std+u
        if self.flag:
            lmbda = self.parameter_dict["lambda"]
            return pd.Series(inv_boxcox(data, lmbda), name=self.name)
        else:
            return data


class CatHandlerHighCard:
    def __init__(self, threshold=50, method="log"):
        self.threshold = threshold
        self.categories = list()
        self.method = None
        self.encoder = OneHotEncoder(sparse=False)

    def determine_threshold(self):
        pass

    def fit(self, series):
        data = series.copy()
        self.name = data.name
        value_count_df = data.value_counts()
        self.categories = list(
            value_count_df[value_count_df > self.threshold].keys())
        data[data.isin(self.categories) ==
             False] = "NoneClass"
        self.encoder.fit(pd.DataFrame(data))

    def transform(self, series):
        data = series.copy()
        data[data.isin(self.categories) ==
             False] = "NoneClass"

        data = self.encoder.transform(pd.DataFrame(data))
        return pd.DataFrame(data, columns=self.encoder.get_feature_names([self.name])).reset_index(drop=True)


class IdentityTransform:
    def __init__(self):
        pass

    def fit(self, series):
        pass

    def transform(self, series):
        return series.copy().reset_index(drop=True)


def get_column_names_from_ColumnTransformer(column_transformer):
    col_name = []
    # the last transformer is ColumnTransformer's 'remainder'
    for transformer_in_columns in column_transformer.transformers_:
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline):
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError:  # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names, np.ndarray):  # eg.
            col_name += names.tolist()
        elif isinstance(names, list):
            col_name += names
        elif isinstance(names, str):
            col_name.append(names)
    return col_name
