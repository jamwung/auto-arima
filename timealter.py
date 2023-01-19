import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
from tqdm.notebook import tqdm

from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

class TSError(ValueError):
    pass

class TimeAlter:
    """
    Forecast a time series using ARIMA from statsmodels.
    
    The best estimator is chosen based on a grid search on the `order` of
    ARIMA.
    
    Parameters
    ----------
    ts : DataFrame of shape (n_samples, 1)
        Dataframe where the index is a valid datetime-like object and the
        column contains the values to be forecasted

    h : int, default=1
        Number of steps to forecast (forecast horizon)

    cv : int, default=None
        Number of validation sets of size `h` to use for time series k-fold
        cross-validation

    test_sets : int, default=1
        Number of test sets of size `h` to use for holdout testing

    freq : str, default='D'
        Valid pandas frequency alias, e.g., {'D', 'M', 'Y'}

    Attributes
    ----------
    time_series_ : DataFrame of shape (n_samples, 1)
        Input time series
    
    transformed_series_ : DataFrame of shape (n_samples, 1)
        Transformed or differenced time series based on `time_series_`
        attribute
    
    cv_splits : list of n_splits
        contains tuples (train_index, val_index) for cross-validation
        
    test_splits : list of n_splits
        contains tuples (train_index, val_index) for testing
        
    cv_summary : pandas DataFrame of shape (n_params, 3)
        contains the average mean absolute errors from grid search
        cross-validation
    """
    
    def __init__(self, ts, h=1, cv=None, test_sets=1, freq='D'):
        if type(h) != int:
            raise TSError('Input a valid h.')
        if type(test_sets) != int:
            raise TSError('Input a valid test_sets.')
        if cv is not None:
            if type(cv) != int:
                raise TSError('Input a valid cv.')
        if type(freq) != str:
            raise TSError('Input a valid freq.')
        self.time_series_ = pd.DataFrame(ts)
        self._log = None
        self.transformed_series_ = None
        self.forecast_horizon_ = h
        self.frequency_ = freq
        self.cv_ = cv
        self.test_sets_ = test_sets
        self._check_integrity()

    def __repr__(self):
        return (f'TimeAlter(h={self.forecast_horizon_}, '
                f'freq={self.frequency_})'
               )

    def _check_integrity(self):
        """
        Check whether the input time series has missing values or dates. If
        there are missing dates, concatenate them to the time series.
        """
        index_name = self.time_series_.index.name
        if type(self.time_series_.index) == pd.PeriodIndex:
            func = pd.period_range
        elif type(self.time_series_.index) == pd.DatetimeIndex:
            func = pd.date_range
        else:
            raise TSError('Input DataFrame is not a valid time series.')
        dt_range = func(start=self.time_series_.index[0],
                        end=self.time_series_.index[-1],
                        freq=self.frequency_)
        self.date_range_ = dt_range
        
        if len(self.time_series_) != len(dt_range):
            na_idx = np.setdiff1d(dt_range, self.time_series_.index)
            self.time_series_ = (
                pd.concat([self.time_series_, pd.DataFrame(index=na_idx)])
                .sort_index()
            )
            self.time_series_.index.name = index_name
            print('Time series has missing dates. Index has been updated.')
        missing = self.time_series_.squeeze().isna().sum()
        if missing != 0:
            print(f'There are {missing} missing values.')
    
    def impute(self, method='ffill'):
        """
        Impute the missing values of the input time series using the 
        method specified.
        """
        if self.time_series_.squeeze().isna().sum() == 0:
            print('There are no missing values to fill.')
        else:
            if method == 'ffill':
                self.time_series_.ffill(inplace=True)
            elif method == 'bfill':
                self.time_series_.bfill(inplace=True)
            elif method == 'interpolate':
                self.time_series_.interpolate(inplace=True)
            print('Missing values have been imputed.')
    
    def split(self):
        """
        Split the time series to train, validation, and test sets. Uses
        `TimeSeriesSplit` from `sklearn`.
        """
        if self.transformed_series_ is not None:
            ts = self.transformed_series_
        else:
            ts = self.time_series_

        self.test_splits = list(
            TimeSeriesSplit(n_splits=self.test_sets_,
                            test_size=self.forecast_horizon_)
            .split(ts)
        )
        if self.cv_ is not None:
            self.cv_splits = list(
                TimeSeriesSplit(n_splits=self.cv_,
                                test_size=self.forecast_horizon_)
                .split(self.test_splits[0][0])
            )

    def get_baseline(self, m=None):
        """
        Get baseline forecasts using the naive and seasonal naive models.
        """
        ts = self.time_series_.squeeze()
        naive_mae = []
        snaive_mae = []
        for train_idx, test_idx in self.test_splits:
            y_actual = ts.iloc[test_idx]
            naive_forecast = pd.Series(
                np.repeat(ts.iloc[train_idx[-1]],
                          self.forecast_horizon_),
                index=y_actual.index
            )
            
            if m > self.forecast_horizon_:
                snaive_forecast = pd.Series(
                    ts.iloc[train_idx]
                    .iloc[-m:-m+self.forecast_horizon_].to_numpy(),
                    index=y_actual.index
                )
            elif m == self.forecast_horizon_:
                snaive_forecast = pd.Series(
                    ts.iloc[train_idx].iloc[-m:].to_numpy(),
                    index=y_actual.index
                )
            naive_mae.append(mean_absolute_error(y_actual, naive_forecast))
            snaive_mae.append(mean_absolute_error(y_actual, snaive_forecast))
        self.baseline = dict(
            naive_mae=np.mean(naive_mae),
            snaive_mae=np.mean(snaive_mae)
        )
        return self.baseline

    def adf_test(self, ts, alpha=0.05):
        """
        Check the stationarity of a time series `ts` using adfuller test.
        """
        res = adfuller(ts)
        print('ADF = %f' % res[0])
        print('p-value = %f' % res[1])
        if res[1] < alpha:
            print(f'The time series is probably stationary.')
        else:
            print(f'The time series is probably NOT stationary.')
    
    def eda(self, alpha=0.05, lags=50):
        """
        Plot the time series and its autocorrelation plot.
        """
        self.adf_test(self.time_series_, alpha=alpha)
        fig, ax = plt.subplots(2, 1, figsize=(6.4*2.5, 4.8*2))
        fig.tight_layout(pad=5)
        self.time_series_.plot(ax=ax[0], lw=2, color='midnightblue')
        ax[0].set_title('Time Series Plot')
        ax[0].autoscale()
        
        plot_acf(self.time_series_, ax=ax[1], alpha=alpha, lw=2,
                 lags=lags, zero=False,
                 color='midnightblue')
        ax[1].autoscale()
        return ax
    
    def transform(self, log=False, diff=None):
        """
        Transform the time series using either log-transformation,
        differencing, or both.
        """
        if log:
            self._log = log
            self.transformed_series_ = np.log(self.time_series_)
        else:
            self.transformed_series_ = self.time_series_.copy()
        
        if diff is not None:
            if type(diff) == int:
                self.transformed_series_ = (self.time_series_
                                            .diff(diff)[diff:])
                self._diffs = [(diff, self.time_series_)]
            elif type(diff) == list:
                ts_diffs = [self.time_series_]
                for d_ in diff:
                    self.transformed_series_ = (self.transformed_series_
                                                .diff(d_).iloc[d_:])
                    ts_diffs.append(self.transformed_series_)
                self._diffs = list(zip(diff, ts_diffs[:-1]))
            else:
                raise TSError(
                    f'Type {type(diff)} is not a valid differencing'
                    f'lag. Input either {int} or {list} type.'
                )
        
        self.split()
        
        self.adf_test(self.transformed_series_)
        return self.transformed_series_
    
    def inverse_transform(self, forecast):
        """
        Perform the inverse transformation of the `transform` method.
        """
        y_pred = forecast.copy()
        for d_, transformed_series in self._diffs:
            y_pred += transformed_series.shift(d_).loc[y_pred.index].squeeze()
        if self._log:
            y_pred = np.exp(y_pred)
        
        return y_pred

    def run_arima(self, ts, split, order, plot=False):
        """
        Test an ARIMA model on the time series.
        """
        mae = []
        if plot:
            fig, ax = plt.subplots(figsize=(6.4*2.5, 4.8))
            y_val_list = []
        for train_index, val_index in tqdm(split):
            x_train = ts.iloc[train_index]
            offset = sum([d_ for d_, _ in self._diffs])
            y_val = self.time_series_.iloc[offset:].iloc[val_index]
            
            model = ARIMA(x_train, order=order).fit()
            y_pred = model.forecast(self.forecast_horizon_)

            if self.transformed_series_ is not None:
                forecast = self.inverse_transform(y_pred).squeeze()
            else:
                forecast = y_pred.copy()
            
            if plot:
                y_val_list.append(y_val)
                forecast.plot(ax=ax, color='forestgreen', linestyle='--')
            
            mae.append(mean_absolute_error(y_val, forecast))
        if plot:
            pd.concat(y_val_list).plot(ax=ax, color='midnightblue')
    
        return np.mean(mae)
    
    def arima_cv(self, param_grid=dict(p=[1, 2], d=[0], q=[1, 2])):
        """
        Perform an ARIMA grid search on the time series.
        """
        if self.transformed_series_ is not None:
            ts = self.transformed_series_
        else:
            ts = self.time_series_
        
        grid = list(itertools.product(param_grid['p'], param_grid['d'],
                                      param_grid['q']))
        df_results = pd.DataFrame()
        avg_mae = []
        for pdq in grid:
            cv_mae = self.run_arima(ts, self.cv_splits, pdq)
            avg_mae.append(np.mean(cv_mae))
            
        df_results['order_pdq'] = grid
        df_results['mean_mae'] = avg_mae
        
        ordered_results = df_results.sort_values('mean_mae')
        self.cv_summary = ordered_results
        self.best_arima_ = ordered_results['order_pdq'].iloc[0]
        
        test_mae = self.run_arima(ts, self.test_splits,
                                  self.best_arima_, plot=True)
        self.test_mae = test_mae
        return self.test_mae