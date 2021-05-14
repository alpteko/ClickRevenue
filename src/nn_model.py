import keras
import pandas as pd
from imblearn.over_sampling import SMOTE

class_keras_metric_list = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  #
]


def make_keras_model(output_bias=None):

    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(
            64, activation='relu', kernel_regularizer='l2'),
        keras.layers.Dense(
            1, kernel_regularizer='l2')
    ]
    )
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss=keras.losses.MeanSquaredError(), metrics="mse")
    return model


def train_evaluate_nn(train_tpl,  dev_tpl):
    (train_X_df, train_y) = train_tpl
    (dev_X_df, dev_y) = dev_tpl
    over_sampler = SMOTE(sampling_strategy=0.2, k_neighbors=2)
    train_X_df, train_y = over_sampler.fit_resample(train_X_df, train_y)
    classfier = make_keras_model(0.01)
    classfier.fit(train_X_df, train_y, batch_size=16, epochs=5)
    pred_proba = classfier.predict(dev_X_df).flatten()
    pred_class = (pred_proba > 0.5) + 0
    return pd.DataFrame({"prob": pred_proba,
                        "pred_class": pred_class,
                         "real": dev_y.values})
