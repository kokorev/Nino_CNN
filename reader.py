import os
import numpy as np
import pandas as pd
import netCDF4
import random
from datetime import timedelta


class Reader:
    base_path = './data'
    var1, var2 = 'sohtc300', 'sosstsst'

    nino_fn = 'iersst_nino3.4a_rel.dat'
    pdo_fn = 'ipdo.dat'

    def __init__(self):
        self.y_series = self.get_y_series()

    def get_fn(self, var, opa, be=False):
        fn_templ = 'norm_{}_opa{}{}.nc'
        return os.path.join(self.base_path, fn_templ.format(var, opa, '_be' if be else ''))

    def get_oras_data(self, var, opa, be=False):
        fn = self.get_fn(var, opa, be)
        ds = netCDF4.Dataset(fn)
        var = np.array(ds.variables[var])
        var = var[:, :-2, :]  # throw away Anarctic
        var[var > 1] = 0
        var = np.moveaxis(var, 0, 2)
        time = ds.variables['time_counter']
        dates = netCDF4.num2date(time[:], time.units)
        nc_dti = pd.DatetimeIndex(pd.date_range(
            start=dates[0].strftime('%Y-%m-%d'),
            end=(dates[-1] + timedelta(days=17)).strftime('%Y-%m-%d'),
            freq='M')
        )
        return var, nc_dti

    def get_y_series(self, normalize=True):
        fn = os.path.join(self.base_path, self.nino_fn)
        nino = pd.read_table(fn, comment='#', sep='\ +', header=None)[1]
        nino.index = pd.DatetimeIndex(pd.date_range(start='1854-01-15', end='2020-03-15', freq='M'))
        if normalize:
            nino = (nino - nino.min()) / (nino.max() - nino.min())
        return nino

    def get_xy(self, opa, n_forward, lookback=4, be=False):
        var1, nc_dti1 = self.get_oras_data(self.var1, opa, be)
        var2, nc_dti2 = self.get_oras_data(self.var2, opa, be)
        X, y = [], []
        y_oras = self.y_series[self.y_series.index >= nc_dti1[0]]
        for i in range(lookback, len(nc_dti1) + 1):
            channel_1 = var1[:, :, i - lookback:i]
            channel_2 = var2[:, :, i - lookback:i]
            predicted_date = y_oras.index[i + n_forward]
            val = np.concatenate([channel_1, channel_2], axis=2)  # , channel_3
            X.append(val)
            y_val = y_oras[y_oras.index[i + n_forward]]
            y.append(y_val)
        X = np.array(X)
        y = np.array(y)
        return X, y

    @staticmethod
    def train_test_split(X, y, test_month, seed=32):
        n_test_points = test_month
        if seed is not None:
            random.seed(seed)
        start_point = random.randrange(1, len(y) - n_test_points)
        X_train = np.concatenate([X[:start_point], X[start_point + n_test_points:]])
        X_test = X[start_point:start_point + n_test_points]
        y_train = np.concatenate([y[:start_point], y[start_point + n_test_points:]])
        y_test = y[start_point:start_point + n_test_points]
        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_test_split_tail(X, y, test_month):
        split_point = -test_month
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        return X_train, X_test, y_train, y_test