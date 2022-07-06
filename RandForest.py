import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV,GridSearchCV
from scipy.stats import randint
from sklearn.utils import shuffle

pd.set_option('display.max_columns', 8)


class RandForest:
    def __init__(self):
        self.ml = RandomForestRegressor(n_estimators=510, max_depth=5, min_samples_split=2,
                                        n_jobs=-1, criterion="squared_error",
                                        min_samples_leaf=1, bootstrap=True)
    def defaultForest(self):
        self.ml = RandomForestRegressor(n_estimators=510, max_depth=5, min_samples_split=2,
                                        n_jobs=-1, criterion="squared_error",
                                        min_samples_leaf=1, bootstrap=True)
    def train(self, filepath,isid=True,isst=True):
        frame = pd.read_csv(filepath)
        if isid:
            frame = frame.drop_duplicates(subset=['id'], keep='last')
        #frame = frame[frame['status'] == 0]
        # убираем ненужные столбцы
        #col = ['id','status']#,'skipsN','skipsV']
        x = None
        if isid and isst:
            col = ['id','status']
            x = frame.drop(columns=col).values
        elif isid and not isst:
            col = ['id']
            x = frame.drop(columns=col).values
        elif not isid and isst:
            col = ['status']
            x = frame.drop(columns=col).values
        elif not isid and not isst:
            x = frame.values


        y = frame['status'].values
        # получаем тестовые и тренировочные выборки
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)
        X_normalized = x_train
        # X_normalized = normalize(x_train, norm='l2')
        # обучаем модель
        self.ml.fit(X_normalized, y_train)
        # предсказываем тестовую выборку
        y_pred = self.ml.predict(x_test)
        r2s = r2_score(y_test, y_pred)
        # возврат квадрата коэффициента множественной корреляции
        return r2s
    def getStatsFrameAndModel(self, filepath):
        """
        Finctuon return statustics to Model and DataFrame

        :param filepath:
        :return:basic_param,df_frame
        where
        basic_param = {'MSE': float, 'MAE': float, 'MAPE': float, 'R2scope': float}
        df_frame = 'dR2scope', 'dMSE', 'dMAE', 'dMAPE' in i columns
        """
        frame = pd.read_csv(filepath)
        frame = frame.drop_duplicates(subset=['id'], keep='last')
        col = ['id', 'status']# ,'skipsN', 'skipsV', 'marks']
        frm = frame.drop(columns=col)
        x = frm.values
        y = frame['status'].values
        # получаем тестовые и тренировочные выборки
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)#test_size=0.75
        X_normalized = x_train
        # X_normalized = normalize(x_train, norm='l2')
        # обучаем модель
        self.ml.fit(X_normalized, y_train)
        # id,marks,attestations,debtsL,debtsO,skipsV,skipsN,status
        # marks,attestations,debtsL,debtsO,skipsV,skipsN
        y_pred = self.ml.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        # Средняя абсолютная процентная погрешность (MAPE) регрессионных потерь
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2s = r2_score(y_test, y_pred)
        columns_name_bas = ['MSE', 'MAE', 'MAPE', 'R2scope']
        basic_param = pd.Series([mse, mae, mape, r2s], index=columns_name_bas)
        # basic_param = {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R2scope': r2s}
        df_shuffled = shuffle(pd.DataFrame((x_test))).reset_index(drop=True)
        x_test_tmp = pd.DataFrame((x_test))
        arr = []
        for i in df_shuffled:
            tmp = x_test_tmp.copy()
            tmp[i] = df_shuffled[i]
            pred = self.ml.predict(tmp.values)
            t_r2s = r2_score(y_test, pred)
            t_mse = mean_squared_error(y_test, pred)
            t_mae = mean_absolute_error(y_test, pred)
            t_mape = mean_absolute_percentage_error(y_test, pred)
            arr.append([abs(t_r2s-r2s), abs(t_mse-mse), abs(t_mae-mae), abs(t_mape-mape)])
        columns_name = ['dR2scope', 'dMSE', 'dMAE', 'dMAPE']
        name_index = frm.columns.values.tolist()
        df_frame = pd.DataFrame(arr, index=name_index, columns=columns_name)
        return basic_param, df_frame
    def trySetBestParams(self,filepath):
        """ Very Long """
        frame = pd.read_csv(filepath)
        frame = frame.drop_duplicates(subset=['id'], keep='last')
        col = ['id', 'status']
        x = frame.drop(columns=col).values
        y = frame['status'].values
        rng = np.random.RandomState(0)
        clf = RandomForestRegressor(random_state=rng)#, n_jobs=-1)
        param_dist = {
            "max_depth": [3, 5, 20, None],
            "max_features": ["auto", 0.5, 0.9],
            "min_samples_split": [2, 5, 20],
            "min_samples_leaf": [1, 3, 5, 20, 50],
            "n_estimators": [10, 30, 70, 100, 200, 500, 700, 1000],
        }
        # GridSearchCV
        # rsh = HalvingRandomSearchCV(
        rsh = GridSearchCV(
            estimator=clf, param_grid=param_dist, n_jobs=-1)  # param_distributions=param_dist,factor=2,random_state=rng
        rsh.fit(x, y)
        best = rsh.best_params_
        print("best param = ", best)
        self.ml = RandomForestRegressor(n_estimators=best["n_estimators"],
                                        max_depth=best["max_depth"],
                                        min_samples_split=best["min_samples_split"],
                                        max_features=best["max_features"],
                                        n_jobs=-1, criterion="squared_error",
                                        min_samples_leaf=best["min_samples_leaf"], bootstrap=True)
    def getStatsFrame(self, filepath):
        """Returns DataFrame (percentOfZeros,mean,std) in i column"""
        frame = pd.read_csv(filepath)
        frame = frame.drop_duplicates(subset=['id'], keep='last')
        col = ['id', 'status']# ,'skipsN', 'skipsV', 'marks']
        frm = frame.drop(columns=col)
        #  frm = frm[(frame.T > 0.001).any()]
        #frm = frm[frm['marks'] > 0]
        arr_row = []
        for column_name in frm.columns:
            column: pd.Series = frm[column_name]
            # Get the count of Zeros in column
            count = (column < 0.001).sum()
            count = count/len(column)*100 if len(column) > 0 else 0
            # Get the mean  in column
            # Get the std  in column
            arr_row.append((int(count),column.mean(),column.std()))
        column_nm =[ 'percent_of_zeros', 'mean', 'std']
        name_index = frm.columns.values.tolist()
        return pd.DataFrame(arr_row, columns=column_nm,index=name_index)
    def compStudentMembership(self,arr):
        x = [arr[1:-1]]
        x = np.array(x,dtype=float)
        y = self.ml.predict(x)
        #p = lgs(y)
        return y[0]
    def saveParams(self):
        ''' Сохранить модель в файл'''
        s = pickle.dumps(self.ml)
        dump(s,'weight.joblib')
    def loadParams(self):
        '''  Загрузить модель из файла  '''
        s = load('weight.joblib')
        self.ml = pickle.loads(s)
