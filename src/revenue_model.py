import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import pearsonr
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from src.transformer import ConTransform


class RevenueModel:
    def __init__(self,  purchase_model, revenue_model, transformer_list, pca=None):
        self.purchase_model = purchase_model
        self.revenue_model = revenue_model
        self.transformer_list = transformer_list
        self.pca = pca
        self.revenue_transform = ConTransform()

    def train(self, data_df, sampling="under", sampling_strategy=0.5):
        self.train_data = data_df.copy()
        self.prepare_data(data_df)
        self._train(sampling=sampling, sampling_strategy=sampling_strategy)
        self.evaluate()

    def _train(self, sampling="under", sampling_strategy=0.5):
        if sampling == "over":
            sampler = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=2)
            p_train_df, p_train_y = sampler.fit_resample(
                self.p_train_df, self.p_train_y)
            self.purchase_model.fit(p_train_df, p_train_y)
        elif sampling == "under":
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
            p_train_df, p_train_y = sampler.fit_resample(
                self.p_train_df, self.p_train_y)
            self.purchase_model.fit(p_train_df, p_train_y)

        else:
            self.purchase_model.fit(self.p_train_df, self.p_train_y)

    def evaluate(self):
        self.revenue_model.fit(self.r_train_df, self.r_train_y)
        # Predict probability
        self.p_dev_prediction = self.purchase_model.predict(self.p_dev_df)
        self.p_dev_prediction_probs = self.purchase_model.predict_proba(
            self.p_dev_df)[:, 1]
        # Evaluate Purchase
        self.evaluate_purchase(self.p_dev_prediction,
                               self.p_dev_prediction_probs)
        # Revenue Model Prediction
        r_dev_prediction = self.revenue_model.predict(self.p_dev_df)
        self.r_dev_prediction = self.revenue_transform.inv_transform(
            r_dev_prediction)
        # Evaluate Purchase  as if we had a perfect classifier.
        self.evaluate_perfect_revenue(self.r_dev_prediction)
        # Purchase Model Probability
        self.p_dev_probs = self.purchase_model.predict_proba(self.p_dev_df)
        self.r_dev_prediction *= self.p_dev_probs[:, 1]
        # Evaluate Purchase  in the real setting. After probability applied.
        self.evaluate_revenue(self.r_dev_prediction)

    def get_report(self):
        self.report = pd.DataFrame({"f1_micro": [self.micro_f1],
                                    "f1_macro": [self.macro_f1],
                                    "f1_weighted": [self.w_f1],
                                    "auc": [self.roc],
                                    "perfect_r2": [self.perfect_r2_score],
                                    "perfect_mape": [self.perfect_mape],
                                    "perfect_mae": [self.perfect_mae],
                                    "perfect_pearson": [self.perfect_pearson],
                                    "r2": [self.r2_score],
                                    "mape": [self.mape],
                                    "mae": [self.mae],
                                    "pearson": [self.pearson]})
        return self.report.copy()

    def vis(self):
        print(classification_report(
            y_pred=self.p_dev_prediction, y_true=self.p_dev_y))

    def confusion_matrix(self):
        return confusion_matrix(y_pred=self.p_dev_prediction, y_true=self.p_dev_y)

    def evaluate_purchase(self, p_dev_prediction, p_dev_prediction_probs):
        self.macro_f1 = f1_score(
            y_pred=p_dev_prediction, y_true=self.p_dev_y, average='macro')
        self.micro_f1 = f1_score(
            y_pred=p_dev_prediction, y_true=self.p_dev_y, average='micro')
        self.w_f1 = f1_score(
            y_pred=p_dev_prediction, y_true=self.p_dev_y, average='weighted')
        self.roc = roc_auc_score(
            y_score=p_dev_prediction_probs, y_true=self.p_dev_y)

    def evaluate_perfect_revenue(self, r_dev_prediction):
        mask = self.r_dev_y > 0
        self.perfect_r2_score = r2_score(
            y_pred=r_dev_prediction[mask], y_true=self.r_dev_y[mask])
        self.perfect_mape = mean_absolute_percentage_error(
            y_pred=r_dev_prediction[mask], y_true=self.r_dev_y[mask])
        self.perfect_mae = mean_absolute_error(
            y_pred=r_dev_prediction[mask], y_true=self.r_dev_y[mask])
        self.perfect_pearson = pearsonr(r_dev_prediction[mask],
                                        self.r_dev_y[mask])[0]

    def evaluate_revenue(self, r_dev_prediction):
        self.r2_score = r2_score(
            y_pred=r_dev_prediction, y_true=self.r_dev_y)
        self.mape = mean_absolute_percentage_error(
            y_pred=r_dev_prediction, y_true=self.r_dev_y)
        self.mae = mean_absolute_error(
            y_pred=r_dev_prediction, y_true=self.r_dev_y)
        self.pearson = pearsonr(r_dev_prediction,
                                self.r_dev_y)[0]

    def predict(self, test_df):
        # Test Prediction
        feature_list = list()
        for key, transformer in self.transformer_list:
            feature_list.append(transformer.transform(test_df[key]))
        test_feature = pd.concat(feature_list, 1)
        revenu_prediction = self.revenue_model.predict(test_feature)
        revenu_prediction = self.revenue_transform.inv_transform(
            revenu_prediction)
        # Purchase Model Probability
        purchase_prediction = self.purchase_model.predict_proba(test_feature)
        final_prediction = revenu_prediction * purchase_prediction[:, 1]
        return final_prediction

    def prepare_data(self, data_df):
        # Transformers
        # Split Data

        revenue_df = data_df.loc[data_df.purchase == 1, :]
        non_revenue_df = data_df.loc[data_df.purchase == 0, :]
        r_train_df, r_dev_df = train_test_split(revenue_df, test_size=0.2)
        p_train_df, p_dev_df = train_test_split(non_revenue_df, test_size=0.2)
        # Purchase Classification Dataset
        # Train
        p_train_df = pd.concat([p_train_df, r_train_df])
        p_train_y = p_train_df.purchase.copy()
        # Dev
        p_dev_df = pd.concat([p_dev_df, r_dev_df])
        p_dev_y = p_dev_df.purchase.copy()
        # Revenue Regression Dataset
        # train
        r_train_y = r_train_df.net_revenue.copy()
        r_dev_y = p_dev_df.net_revenue.copy()
        # Fit Transforms
        for key, transformer in self.transformer_list:
            #  print(key)
            transformer.fit(p_train_df[key])
        self.revenue_transform.fit(r_train_y)
        p_train_list = list()
        p_dev_list = list()
        r_train_list = list()
        r_dev_list = list()
        for key, transformer in self.transformer_list:
            p_train_list.append(transformer.transform(p_train_df[key]))
            p_dev_list.append(transformer.transform(p_dev_df[key]))
            r_train_list.append(transformer.transform(r_train_df[key]))
            r_dev_list.append(transformer.transform(r_dev_df[key]))

        self.p_train_df = pd.concat(p_train_list, 1)
        self.p_train_y = p_train_y.reset_index(drop=True)
        self.p_dev_df = pd.concat(p_dev_list, 1)
        self.p_dev_y = p_dev_y.reset_index(drop=True)
        self.r_train_df = pd.concat(r_train_list, 1)
        self.r_train_y = self.revenue_transform.transform(r_train_y)
        self.r_dev_df = pd.concat(r_dev_list, 1)
        self.r_dev_y = r_dev_y.reset_index(drop=True)
