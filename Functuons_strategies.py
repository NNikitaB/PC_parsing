import pandas as pd
import random
from sklearn.preprocessing import normalize, QuantileTransformer, PolynomialFeatures
from sklearn.ensemble import IsolationForest
import RandForest

# numeric_strategies


class StrategyForNumeric:
    @staticmethod
    def drop_duplicates_all(file_pandas: pd.DataFrame):
        """delete duplicates in str"""
        # file_pandas = file_pandas.drop(file_pandas.columns[[0, 1, 2, 3, 4]], axis=1)
        file_pandas = file_pandas.drop_duplicates(subset=None, keep=False)
        print("dddddddddddddddd")
        print(file_pandas)
        return file_pandas

    @staticmethod
    def drop_duplicates_rand_first(file_pandas: pd.DataFrame):
        """delete duplicates random row in str"""
        # file_pandas = file_pandas.drop(file_pandas.columns[[0, 1, 2, 3, 4]], axis=1)
        file_pandas = file_pandas.drop_duplicates(subset=random.choice(file_pandas.columns), keep="first")
        print(file_pandas)
        return file_pandas

    @staticmethod
    def drop_duplicates_rand_last(file_pandas: pd.DataFrame):
        """delete duplicates random row in str"""
        # file_pandas = file_pandas.drop(file_pandas.columns[[0, 1, 2, 3, 4]], axis=1)
        file_pandas = file_pandas.drop_duplicates(subset=random.choice(file_pandas.columns), keep="last")
        print(file_pandas)
        return file_pandas

    @staticmethod
    def drop_duplicates_rand_all(file_pandas: pd.DataFrame):
        """delete duplicates random row in str"""
        # file_pandas = file_pandas.drop(file_pandas.columns[[0, 1, 2, 3, 4]], axis=1)
        file_pandas = file_pandas.drop_duplicates(subset=random.choice(file_pandas.columns), keep=False)
        print(file_pandas)
        return file_pandas

    @staticmethod
    def drop_anomalies(file_pandas: pd.DataFrame):
        """delete duplicates random row in str"""
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
        ll = lm.train("course_33.csv")
        print(ll)
        ll = lm.train("file_name.csv",isid=False,isst=True)
        print(ll)

        return file_pandas

    @staticmethod
    def del_no_numeric(file_pandas: pd.DataFrame):
        """delete не числовые знацения"""

    @staticmethod
    def is_numeric_frame(file_pandas: pd.DataFrame):
        """                 """

    @staticmethod
    def transform_polynom_features(file_pandas: pd.DataFrame):
        poly = PolynomialFeatures(2)
        arr = poly.fit_transform(file_pandas)
        file_pandas = pd.DataFrame(arr)
        return file_pandas


def get_array_strategies():
    array_strategies = [
        StrategyForNumeric.drop_anomalies,
        StrategyForNumeric.drop_duplicates_all,
        StrategyForNumeric.drop_duplicates_rand_first,
        StrategyForNumeric.drop_duplicates_rand_last,
        StrategyForNumeric.drop_duplicates_rand_all,
        StrategyForNumeric.drop_anomalies,

                        ]
    return array_strategies

    # 3 стратегий написать
    # 12 стратегий написать
    # 12 стратегий написать
    # 12 стратегий написать
    # 12 стратегий написать
    # записать в репозиторий
    # обернуть в модуть препроцесссинг в фастапи FormData
