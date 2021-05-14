import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(path):
    total_df = pd.read_csv(path)
    total_df.rename(columns=lambda x: x.strip(), inplace=True)
    date_process(total_df)
    total_df.loc[:, "cat1"] = total_df["cat1"].apply(lambda x: x.strip())
    total_df.loc[:, "cat2"] = total_df["cat2"].apply(lambda x: x.strip())
    total_df.loc[:, "cat3"] = total_df["cat3"].apply(lambda x: x.strip())
    if "net_revenue" in total_df:
        total_df.loc[:, "purchase"] = total_df["net_revenue"].ge(0.01) + 0
    return total_df.copy()


def date_process(df):
    df.loc[:, "date"] = pd.to_datetime(df.day_id, format="%Y%m%d")
    df.loc[:, "weekday"] = df.date.dt.dayofweek
    df.loc[:, "week"] = df.date.dt.week
    df.loc[:, "month"] = df.date.dt.month


def split_dataset(df):
    new_df = df.copy()
    new_df.loc[:, "purchase"] = new_df.net_revenue.ge(1e-4) + 0
    revenue_df = new_df.loc[new_df.purchase == 1, :].copy()
    return new_df.copy(), revenue_df.copy()


def high_car_cat_handle(data_df, keys=["cat1", "cat2", "cat3"], threshold=40):
    for key in keys:
        value_count_df = data_df[key].value_counts()
        valid_keys = value_count_df[value_count_df < threshold].keys()
        data_df.loc[data_df[key].isin(valid_keys), key] = "None-Classified"


def prepare_data(purchase_df, tansformer_list, is_pca=True):
    # Transformers
    # Split Data
    revenue_df = purchase_df.loc[purchase_df.purchase == 1, :]
    non_revenue_df = purchase_df.loc[purchase_df.purchase == 0, :]
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
    r_dev_y = r_dev_df.net_revenue.copy()
    # Fit Transforms
    for key, transformer in tansformer_list:
        print(key)
        transformer.fit(p_train_df[key])
    p_train_list = list()
    p_dev_list = list()
    r_train_list = list()
    r_dev_list = list()
    for key, transformer in tansformer_list:
        p_train_list.append(transformer.transform(p_train_df[key]))
        p_dev_list.append(transformer.transform(p_dev_df[key]))
        r_train_list.append(transformer.transform(r_train_df[key]))
        r_dev_list.append(transformer.transform(r_dev_df[key]))
    return (
        (pd.concat(p_train_list, 1), p_train_y),
        (pd.concat(p_dev_list, 1),  p_dev_y),
        (pd.concat(r_train_list, 1), r_train_y),
        (pd.concat(r_dev_list, 1), r_dev_y)
    )
