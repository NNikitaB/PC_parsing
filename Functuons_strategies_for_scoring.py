import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import RandForest


def drop_duplicates_all(file_pandas: pd.DataFrame):
    """drop_duplicates_all
    delete duplicates in str
    :type file_pandas: pd.DataFrame
    """
    dd = file_pandas.drop_duplicates(subset=None, keep=False)
    return file_pandas


def drop_duplicates_rand_first(file_pandas: pd.DataFrame):
    """
    drop_duplicates_rand_first
    delete duplicates random row in str
    """
    file_pandas = file_pandas.drop_duplicates(subset=random.choice(file_pandas.columns), keep="first")
    return file_pandas


def drop_duplicates_rand_last(file_pandas: pd.DataFrame):
    """
    drop_duplicates_rand_last
    delete duplicates random row in str
    """
    file_pandas = file_pandas.drop_duplicates(subset=random.choice(file_pandas.columns), keep="last")
    return file_pandas


def drop_duplicates_rand_all(file_pandas: pd.DataFrame):
    """
    drop_duplicates_rand_all
    delete duplicates random row in str
    """
    file_pandas = file_pandas.drop_duplicates(subset=random.choice(file_pandas.columns), keep=False)
    return file_pandas


def drop_anomalies(file_pandas: pd.DataFrame):
    """
    drop_anomalies
    delete duplicates random row in str
    """
    model = IsolationForest(n_estimators=50, max_samples='auto', contamination='auto', max_features=1.0)
    model.fit(file_pandas)
    anomaly = 'anomaly'
    file_pandas[anomaly] = model.predict(file_pandas)
    file_pandas = file_pandas.loc[file_pandas[anomaly] == 1]
    file_pandas = file_pandas.drop(columns=anomaly)
    return file_pandas

# test
def test_drop_anomalies(file_pandas: pd.DataFrame):
    """
    test_drop_anomalies
    delete duplicates random row in str
    """
    zp = zip(file_pandas.dtypes, file_pandas.columns)
    zp = list(filter(lambda x: x[0] == 'int64' or x[0] == 'float64', zp))
    zp = [i[1] for i in zp]
    file_pandas = file_pandas[zp]
    model = IsolationForest(n_estimators=50, max_samples='auto', contamination='auto', max_features=1.0)
    model.fit(file_pandas)
    anomaly = 'anomaly'
    file_pandas[anomaly] = model.predict(file_pandas)
    file_pandas = file_pandas.loc[file_pandas[anomaly] == 1]
    file_pandas = file_pandas.drop(columns=anomaly)
    print(file_pandas)

    lm = RandForest.RandForest()
    ll = lm.train("course_33c.csv")
    print(ll)
    ll = lm.train("file_name.csv",isid=False,isst=True)
    print(ll)
    return file_pandas


def del_local_outlier_factor(file_pandas: pd.DataFrame):
    """
    del_local_outlier_factor
    LOF
    """
    file_pandas = del_no_numeric(file_pandas)
    model = LocalOutlierFactor()
    model.fit(file_pandas)
    anomaly = 'anomaly'
    file_pandas[anomaly] = model.fit_predict(file_pandas)
    file_pandas = file_pandas.loc[file_pandas[anomaly] == 1]
    file_pandas = file_pandas.drop(columns=anomaly)
    return file_pandas


def del_3sigma(file_pandas: pd.DataFrame):
    """
      del_3sigma
      del 3 sigma из sqrt(len(column) < sigma(column)
      """
    scaler = StandardScaler()
    scaler.fit(file_pandas)
    tmp_file_pandas = scaler.transform(file_pandas)
    tmp_file_pandas = pd.DataFrame(tmp_file_pandas)
    sigmas3 = scaler.var_ * 3
    clmn = len(tmp_file_pandas.columns)
    clmn_sqrt = np.sqrt(clmn)
    lst = []
    for index in tmp_file_pandas.iterrows():
        arr = list(index[1].array)
        tmp = np.abs(arr) > sigmas3
        # кол-во превышений 3 сигма в строке
        tmp = np.sum(tmp)
        if tmp > clmn_sqrt:
            lst.append(index[0])
    file_pandas = file_pandas.drop(labels=lst)
    return file_pandas


def del_no_numeric(file_pandas: pd.DataFrame):
    """delete non-numeric futures"""
    zp = zip(file_pandas.dtypes, file_pandas.columns)
    zp = list(filter(lambda x: x[0] == np.int64 or x[0] == np.float64, zp))
    zp = [i[1] for i in zp]
    file_pandas = file_pandas[zp]
    return file_pandas


def is_numeric_frame(file_pandas: pd.DataFrame):
    """  check frame to numbers """
    for i in file_pandas.dtypes:
        if i != np.int64 or i != np.float64:
            return False
    return True


def get_array_strategies():
    array_strategies = [
        drop_duplicates_all,
        drop_duplicates_rand_first,
        drop_duplicates_rand_last,
        drop_duplicates_rand_all,
        drop_anomalies,
        del_3sigma,
        del_local_outlier_factor,
    ]
    return array_strategies

    # 5 стратегий
    # 6 hour

    # 2 стратегий
    # 2 hour

    # 2 стратегий написать
    # 5 hour

    # обернуть в модуть препроцесссинг в фастапи FormData
    # 5 hour

    # записать в docker
    # 2 hour

    # записать в репозиторий
    # - hour    1
