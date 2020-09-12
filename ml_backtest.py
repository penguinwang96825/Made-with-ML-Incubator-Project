import os
import math
import talib
import joblib
import warnings
import datetime
import shap
import plotly
import classifier
import visualiser
import numpy as np
import pandas as pd
import yfinance as yf
import empyrical as ep
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.offline as py
import plotly.tools as tls
import plotly.graph_objects as go
import xgboost as xgb
import lightgbm as lgb
from config import Config
from xgboost import XGBClassifier
from tqdm import tqdm
from scipy.interpolate import interp1d
from matplotlib.ticker import FuncFormatter
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from timeit import default_timer as timer
from plotly.subplots import make_subplots
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import Trials
from hyperopt import fmin
warnings.filterwarnings("ignore")


def generate_feature(data):
    high = data.High.values
    low = data.Low.values
    close = data.Close.values

    feature_df = pd.DataFrame(index=data.index)
    feature_df["ADX"] = ADX = talib.ADX(high, low, close, timeperiod=14)
    feature_df["ADXR"] = ADXR = talib.ADXR(high, low, close, timeperiod=14)
    feature_df["APO"] = APO = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    feature_df["AROONOSC"] = AROONOSC = talib.AROONOSC(high, low, timeperiod=14)
    feature_df["CCI"] = CCI = talib.CCI(high, low, close, timeperiod=14)
    feature_df["CMO"] = CMO = talib.CMO(close, timeperiod=14)
    feature_df["DX"] = DX = talib.DX(high, low, close, timeperiod=14)
    feature_df["MINUS_DI"] = MINUS_DI = talib.MINUS_DI(high, low, close, timeperiod=14)
    feature_df["MINUS_DM"] = MINUS_DM = talib.MINUS_DM(high, low, timeperiod=14)
    feature_df["MOM"] = MOM = talib.MOM(close, timeperiod=10)
    feature_df["PLUS_DI"] = PLUS_DI = talib.PLUS_DI(high, low, close, timeperiod=14)
    feature_df["PLUS_DM"] = PLUS_DM = talib.PLUS_DM(high, low, timeperiod=14)
    feature_df["PPO"] = PPO = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    feature_df["ROC"] = ROC = talib.ROC(close, timeperiod=10)
    feature_df["ROCP"] = ROCP = talib.ROCP(close, timeperiod=10)
    feature_df["ROCR100"] = ROCR100 = talib.ROCR100(close, timeperiod=10)
    feature_df["RSI"] = RSI = talib.RSI(close, timeperiod=14)
    feature_df["ULTOSC"] = ULTOSC = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    feature_df["WILLR"] = WILLR = talib.WILLR(high, low, close, timeperiod=14)
    feature_df = feature_df.fillna(0.0)

    matrix = np.stack((
        ADX, ADXR, APO, AROONOSC, CCI, CMO, DX, MINUS_DI, ROCR100, ROC,
        MINUS_DM, MOM, PLUS_DI, PLUS_DM, PPO, ROCP, WILLR, ULTOSC, RSI))
    matrix = np.nan_to_num(matrix)
    matrix = matrix.transpose()

    return feature_df, matrix


def triple_barrier(data, ub, lb, max_period, two_class=True):
    """
    Reference from https://www.finlab.tw/generate-labels-stop-loss-stop-profit/

    Args:
        data (:obj: pd.DataFrame):
            Get data from yahoo finance API with columns: `Open`, `High`, `Low`, `Close`, and (optionally) `Volume`.
            If any columns are missing, set them to what you have available,
            e.g. df['Open'] = df['High'] = df['Low'] = df['Close']
        ub (:obj: float):
            Upper bound means profit-taking.
        lb (:obj: float):
            Lower bound means loss-stop.
        max_period (:obj: int):
            Max time to hold the position.
        two_class (:obj: bool):
            Whether or not the binary signal has been generated.

    Returns:
        :obj: pd.DataFrame
            DataFrame object contains four columns.
            triple_barrier_profit, triple_barrier_sell_time, triple_barrier_signal, binary_signal (optional)

    Example::
        data = yf.download("AAPL")
        ret = triple_barrier(data, ub=1.07, lb=0.97, max_period=20, two_class=True)
    """
    def end_price(s):
        return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0]/s[0]

    r = np.array(range(max_period))

    def end_time(s):
        return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period-1)[0]

    price = data.Close
    p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period+1)
    t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period+1)
    t = pd.Series([t.index[int(k+i)] if not math.isnan(k+i) else np.datetime64('NaT')
                   for i, k in enumerate(t)], index=t.index).dropna()

    signal = pd.Series(0, p.index)
    signal.loc[p > ub] = 1.0
    signal.loc[p < lb] = -1.0

    ret = pd.DataFrame({
        'triple_barrier_profit': p,
        'triple_barrier_sell_time': t,
        'triple_barrier_signal': signal})
    ret = ret.fillna(0)
    ret["{}d_returns".format(max_period)] = data.Close.pct_change(periods=max_period).fillna(0)
    sign = lambda x: math.copysign(1, x)

    if two_class:
        binary_list = []
        for ind, row in ret.iterrows():
            if row["triple_barrier_signal"] == 0:
                binary_list.append(sign(row["{}d_returns".format(max_period)]))
            else:
                binary_list.append(row["triple_barrier_signal"])
        ret["binary_signal"] = binary_list
        ret["binary_signal"] = ret["binary_signal"].apply(lambda x: 1.0 if x == 1.0 else 0.0)

    return ret


def absolute_turning_points(data, plot=True):
    '''
    Finds the turning points within an 1D array and returns the indices of the minimum and
    maximum turning points in two separate lists.
    '''
    array = data.Close
    idx_max, idx_min = [], []
    if (len(array) < 3):
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    pbar = tqdm(total=len(array))

    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                if s == FALLING:
                    idx_max.append((begin + i - 1) // 2)
                else:
                    idx_min.append((begin + i - 1) // 2)
            begin = i
            ps = s
        pbar.update(1)
    pbar.close()

    if plot:
        # plt.figure(figsize=(20, 10))
        # plt.plot(array, alpha=0.7)
        # plt.scatter(array.index[idx_min], array.iloc[idx_min], marker="^", label="buy", color="green")
        # plt.scatter(array.index[idx_max], array.iloc[idx_max], marker="v", label="sell", color="red")
        # plt.title("Absolute Turning Points")
        # plt.legend()
        # plt.grid()
        # plt.show()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x= array.index[idx_min], y=array.iloc[idx_min],
            name='buy',
            mode='markers',
            marker_symbol = 5,
            marker_color = 'green'
        )showlegend = True)
        fig.add_trace(go.Scatter(
            x=array.index[idx_max], y = array.iloc[idx_max],
            name='sell',
            mode ='markers',
            marker_symbol = 6,
            marker_color='red'

        )showlegend=True)
        fig.update_traces(mode='markers', marker_line_width=0.7, marker_size=5)
        fig.update_layout(title='Absolute Turning Points',
                  yaxis_zeroline=False, xaxis_zeroline=False)
        fig.show()

        return idx_min, idx_max





def relative_turning_points(data, step_size=10, interpolation_kind='cubic', plot=True):
    '''
    Reference from https://www.quantopian.com/posts/quick-and-dirty-way-to-find-tops-and-bottoms-of-time-series
    Returns Tops and Bottoms of the inputed price series as a tuple
    '''
    array = data.Close
    # Get smoothed curve
    x = np.arange(0,len(array),step_size)
    f = interp1d(x, array.values[::step_size], bounds_error=False, kind=interpolation_kind)

    # Use forward finite difference method to calculate first derivative
    x = np.arange(0,len(array))
    y = f(x)
    dy = [0.0]*len(x)
    for i in range(len(x)-1):
        dy[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    dy[-1] = (y[-1]-y[-2])/(x[-1]-x[-2])

    # Quick and dirty way to get bottoms and tops without calculating 2nd derivative
    bottoms = []
    tops = []
    prev = dy[0]
    pbar = tqdm(total=len(array))

    for i in range(1, len(x)):
        if prev < 0 and dy[i] > 0: bottoms.append(i)
        elif prev > 0 and dy[i] < 0: tops.append(i)
        prev = dy[i]
        pbar.update(1)
    pbar.close()

    if plot:
        # plt.figure(figsize=(20, 10))
        # plt.plot(array, alpha=0.7)
        # plt.scatter(array.index[bottoms], array.iloc[bottoms], marker="^", label="buy", color="green")
        # plt.scatter(array.index[tops], array.iloc[tops], marker="v", label="sell", color="red")
        # plt.title("Relative Turning Points (step size: {})".format(step_size))
        # plt.legend()
        # plt.grid()
        # plt.show()


        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x= array.index[bottoms], y=array.iloc[bottoms],
            name='buy',
            mode='markers',
            marker_symbol = 5,
            marker_color = 'green'
        ),showlegend = True)
        fig.add_trace(go.Scatter(
            x=array.index[tops], y = array.iloc[tops],
            name='sell',
            mode ='markers',
            marker_symbol = 6,
            marker_color='red'

        ),showlegend=True)
        fig.update_traces(mode='markers', marker_line_width=0.7, marker_size=5)
        fig.update_layout(title='Relative Turning Points (step size: {})'.format(step_size),
                  yaxis_zeroline=False, xaxis_zeroline=False)
        fig.show()

    return bottoms, tops


def fixed_time_horizon(data, threshold, look_forward=1, standardized=False, window=None):
    """
    Reference from https://mlfinlab.readthedocs.io/en/latest/labeling/labeling_fixed_time_horizon.html
    Fixed-Time Horizon Labelling Method
    Originally described in the book Advances in Financial Machine Learning, Chapter 3.2, p.43-44.
    Returns 1 if return at h-th bar after t_0 is greater than the threshold, -1 if less, and 0 if in between.

    Args:
        data (:obj: pd.DataFrame):
            Close prices over fixed horizons (usually time bars, but can be any format as long as
            index is timestamps) for a stock ticker.
        threshold (:obj: float or pd.Series):
            When the abs(change) is larger than the threshold, it is labelled as 1 or -1.
            If change is smaller, it's labelled as 0. Can be dynamic if threshold is pd.Series. If threshold is
            a series, threshold.index must match close.index. If threshold is negative, then the directionality
            of the labels will be reversed.
        look_forward (:obj: int):
            Number of ticks to look forward when calculating future return rate. (1 by default)
            If n is the numerical value of look_forward, the last n observations will return a label of NaN
            due to lack of data to calculate the forward return in those cases.
        standardized (:obj: bool):
            Whether returns are scaled by mean and standard deviation.
        window (:obj: int):
            If standardized is True, the rolling window period for calculating the mean and standard
            deviation of returns.

    Returns:
        :obj: np.array
            -1, 0, or 1 denoting whether return for each tick is under/between/greater than the threshold.
            The final look_forward number of observations will be labeled np.nan.
    """
    # Calculate forward price with
    close = data.Close
    forward_return = close.pct_change(periods=look_forward).shift(-look_forward)

    # Warning if look_forward is greater than the length of the series,
    if look_forward >= len(forward_return):
        warnings.warn('look_forward period is greater than the length of the Series. All labels will be NaN.',
                      UserWarning)

    # Adjust by mean and stdev, if desired. Assert that window must exist if standardization is on. Warning if window is
    # too large.
    if standardized:
        assert isinstance(window, int), "when standardized is True, window must be int"
        if window >= len(forward_return):
            warnings.warn('window is greater than the length of the Series. All labels will be NaN.', UserWarning)
        mean = forward_return.rolling(window=window).mean()
        stdev = forward_return.rolling(window=window).std()
        forward_return -= mean
        forward_return /= stdev

    # Conditions for 1, 0, -1
    conditions = [forward_return > threshold, (forward_return <= threshold) & (forward_return >= -threshold),
                  forward_return < -threshold]
    choices = [1, np.nan, -1]
    labels = np.select(conditions, choices, default=np.nan)
    return labels


def generate_label(data, method, ub=1.07, lb=0.97, max_period=20,two_class=True,step_size=10,prediction_delay=5,threshold=.04,look_forward=1, standardized=False, window=5):
    """
    Generate labels for supervised machine learning.

    Args:
        data (:obj: pd.DataFrame):
            Get data from yahoo finance API with columns: `Open`, `High`, `Low`, `Close`, and (optional) `Volume`.
            If any columns are missing, set them to what you have available,
            e.g. df['Open'] = df['High'] = df['Low'] = df['Close']
        method (:obj: str):
            "TBM": Triple Barrier Method
            "ATP": Absolute Turning Point Method
            "RTP": Relative Turning Point Method
            "PDM": Prediction Delay Method
            "FTH": Fixed-Time Horizon Method
        ub (:obj: float, 'optional', defaults to 1.07):
            Parameter for "TBM" method.
            Upper bound means profit-taking.
        lb (:obj: float, 'optional', defaults to 0.97):
            Parameter for "TBM" method.
            Lower bound means loss-stop.
        max_period (:obj: int, 'optional', defaults to 20):
            Parameter for "TBM" method.
            Max time to hold the position.
        two_class (:obj: bool, 'optional', defaults to true):
            Parameter for "TBM" method.
            Whether or not the binary signal has been generated.
        step_size: (:obj: bool, 'optional', defaults to true):
            Parameter for "RTP" method.
            Window size for RTP method.
        prediction_delay: (:obj: int, 'optional', defaults to 5):
            Parameter for "PDM" method.
            If 'prediction_delay' days after the stock price goes up, then assign the label 1.
            If 'prediction_delay' days after the stock price goes down, then assign the label 0.
        threshold (:obj: float or pd.Series):
            Parameter for "FTH" method.
            When the abs(change) is larger than the threshold, it is labelled as 1 or -1.
            If change is smaller, it's labelled as 0. Can be dynamic if threshold is pd.Series. If threshold is
            a series, threshold.index must match close.index. If threshold is negative, then the directionality
            of the labels will be reversed.
        look_forward (:obj: int):
            Parameter for "FTH" method.
            Number of ticks to look forward when calculating future return rate. (1 by default)
            If n is the numerical value of look_forward, the last n observations will return a label of NaN
            due to lack of data to calculate the forward return in those cases.
        standardized (:obj: bool):
            Parameter for "FTH" method.
            Whether returns are scaled by mean and standard deviation.
        window (:obj: int):
            Parameter for "FTH" method.
            If standardized is True, the rolling window period for calculating the mean and standard
            deviation of returns.

    Returns:
        :obj: pd.Series
            Signal series contains 0 or 1.
    """
    convert = lambda x: 1.0 if x == 1.0 else 0.0
    if method == "TBM":
        signal = triple_barrier(data, ub=ub, lb=lb, max_period=max_period, two_class=two_class).binary_signal
        return signal
    elif method == "ATP":
        idx_min, idx_max = absolute_turning_points(data, plot=False)
        signal = pd.Series(np.nan, data.index)
        signal.iloc[idx_max] = -1.0
        signal.iloc[idx_min] = 1.0
        signal = signal.fillna(method="ffill").fillna(1.0)
        signal = signal.map(convert)
        return signal
    elif method == "RTP":
        idx_min, idx_max = relative_turning_points(data, step_size=step_size, interpolation_kind='cubic', plot=False)
        signal = pd.Series(np.nan, data.index)
        signal.iloc[idx_max] = -1.0
        signal.iloc[idx_min] = 1.0
        signal = signal.fillna(method="ffill").fillna(1.0)
        signal = signal.map(convert)
        return signal
    elif method == "PDM":
        data['trend'] = np.where(data.Close.shift(-prediction_delay) > data.Close, 1.0, 0.0)
        data = data.ffill()
        signal = data.trend
        return signal
    elif method == "FTH":
        signal = fixed_time_horizon(data, threshold=threshold, look_forward=look_forward, standardized=standardized, window=window)
        signal = pd.Series(signal).fillna(method="ffill").fillna(1.0)
        signal = signal.map(convert)
        return signal


class MLBacktest:
    """
    Backtest a particular (parameterized) strategy on particular data.
    Upon initialization, call method `MLBacktest.run` to run a backtest
    instance, and optimise with HyperOpt.
    """
    def __init__(self, data, strategy, cash=1000.0, fee=0.002):
        """
        Reference from https://gist.github.com/StockBoyzZ/396d48be23fd479a5ca62362b1bc8dc7#file-strategy_test-py
        Reference from https://github.com/kernc/backtesting.py/blob/1512f0e4cd483d7c0c00b6ad6953ca28322b3b7c/backtesting/backtesting.py

        Args:
            data (:obj: pd.DataFrame):
                Get data from yahoo finance API with columns: `Open`, `High`, `Low`, `Close`, and (optionally) `Volume`.
                If any columns are missing, set them to what you have available,
                e.g. df['Open'] = df['High'] = df['Low'] = df['Close']
            strategy (:obj: str):
                Machine learning model from sklearn.
            cash (:obj: float):
                Initial cash to start with.
            fee (:obj: float):
                Commission ratio.
        """
        self.data = data
        self.strategy = strategy
        self.cash = cash
        self.fee = fee

        self.dtrain = data.loc[:Config.TRAIN_VALID_SPLIT_DATE]
        self.dtest = data.loc[Config.TRAIN_VALID_SPLIT_DATE:]
        self.X_train_df, self.X_train = generate_feature(self.dtrain)
        self.X_test_df, self.X_test = generate_feature(self.dtest)

    def run(self, plot=True, stats=True):
        """
        Reference from https://gist.github.com/StockBoyzZ/396d48be23fd479a5ca62362b1bc8dc7#file-strategy_test-py
        Reference from https://github.com/kernc/backtesting.py/blob/1512f0e4cd483d7c0c00b6ad6953ca28322b3b7c/backtesting/backtesting.py

        Args:
            plot (:obj: bool):
                Whether or not the performance has been plotted.
            stats (:obj: bool):
                Whether or not the statistics result has been calculated.
        """
        # Initialization
        data = self.dtest
        cash = self.cash
        fee = self.fee

        # Generate label
        X_train = self.X_train
        X_test = self.X_test
        y_train = generate_label(self.dtrain, method=Config.LEBELLING_METHOD)
        y_test = generate_label(self.dtest, method=Config.LEBELLING_METHOD)

        # HyperOpt
        if self.strategy == "LGBM":
            clf, y_pred = classifier.lgbm_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "XGB":
            clf, y_pred = classifier.xgb_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "CAT":
            clf, y_pred = classifier.cat_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "RIDGE":
            clf, y_pred = classifier.ridge_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "MLP":
            clf, y_pred = classifier.mlp_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "SGD":
            clf, y_pred = classifier.sgd_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "SVM":
            clf, y_pred = classifier.svm_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "KNN":
            clf, y_pred = classifier.knn_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "GNB":
            clf, y_pred = classifier.gnb_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "DTC":
            clf, y_pred = classifier.dtc_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "ADA":
            clf, y_pred = classifier.ada_opt(X_train, y_train, X_test, y_test)
        elif self.strategy == "GBC":
            clf, y_pred = classifier.gbc_opt(X_train, y_train, X_test, y_test)

        data["binary_signal"] = y_pred
        data["binary_signal"] = data["binary_signal"].apply(lambda x: 1.0 if x == 1.0 else 0.0)
        data['status'] = data.binary_signal.shift(1).fillna(0)
        data['buy_price'] = data.Open[np.where((data.status == 1.0) & (data.status.shift(1) == 0.0))[0]]
        data['sell_price'] = data.Open[np.where((data.status == 0.0) & (data.status.shift(1) == 1.0))[0]]
        data = data.fillna(0.0)

        # Calculate trade return and net trade return
        buy_cost = np.array(data.buy_price[data.buy_price != 0])
        sell_price = np.array(data.sell_price[data.sell_price != 0])
        if len(buy_cost) > len(sell_price) :
            buy_cost = buy_cost[:-1]
        trade_return = sell_price / buy_cost - 1
        net_trade_return = trade_return - fee

        # Put trade return and net trade return into dataframe
        data["trade_return"] = 0.0
        data["net_trade_return"] = 0.0
        sell_dates = data.sell_price[data.sell_price != 0].index
        data.loc[sell_dates, "trade_return"] = trade_return
        data.loc[sell_dates, "net_trade_return"] = net_trade_return

        # Plot performance for every strategies
        data["open_daily_return"] = data.Open / data.Open.shift(1) - 1
        data["strategy_return"] = data.status.shift(1) * data.open_daily_return
        data["strategy_net_return"] = data.strategy_return
        data.loc[sell_dates, "strategy_net_return"] = data.loc[sell_dates, "strategy_net_return"] - fee
        data = data.fillna(0.0)
        data['buy_and_hold_equity'] = (data.open_daily_return + 1).cumprod() * cash
        data['strategy_equity'] = (data.strategy_return + 1).cumprod() * cash
        data['strategy_net_equity'] = (data.strategy_net_return + 1).cumprod() * cash

        def simple_drawdown(return_series: pd.Series, cash=1000):
            """
            Args:
                return_series (:obj: pd.DataFrame):

            Returns:
                wealth (:obj: pd.DataFrame)
                peaks (:obj: pd.DataFrame)
                drawdown (:obj: pd.DataFrame)
            """
            wealth_index = cash*(return_series+1).cumprod()
            previous_peak = wealth_index.cummax()
            drawdowns = (wealth_index-previous_peak)/previous_peak
            return pd.DataFrame({
                "wealth": wealth_index,
                "peaks": previous_peak,
                "drawdown": drawdowns
            })

        def _compute_drawdown_duration_peaks(dd: pd.Series):
            iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
            iloc = pd.Series(iloc, index=dd.index[iloc])
            df = iloc.to_frame('iloc').assign(prev=iloc.shift())
            df = df[df['iloc'] > df['prev'] + 1].astype(int)
            # If no drawdown since no trade, avoid below for pandas sake and return nan series
            if not len(df):
                return (dd.replace(0, np.nan),) * 2
            df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
            df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
            df = df.reindex(dd.index)
            return df['duration'], df['peak_dd']

        def _data_period(index):
            """Return data index period as pd.Timedelta"""
            values = pd.Series(index[-100:])
            return values.diff().median()

        def _round_timedelta(value, _period=_data_period(data.index)):
            if not isinstance(value, pd.Timedelta):
                return value
            resolution = getattr(_period, 'resolution_string', None) or _period.resolution
            return value.ceil(resolution)

        if stats:
            s = pd.Series(dtype=object)
            s.loc['Start'] = data.index[0]
            s.loc['End'] = data.index[-1]
            s.loc['Duration'] = s.End - s.Start
            s.loc['Equity Final [$]'] = data.strategy_net_equity[-1]
            s.loc['Equity Peak [$]'] = data.strategy_net_equity.max()
            s.loc['Return [%]'] = (data.strategy_equity[-1] - data.strategy_equity[0]) / data.strategy_equity[0] * 100
            s.loc['Net Return [%]'] = (data.strategy_net_equity[-1] - data.strategy_net_equity[0]) / data.strategy_net_equity[0] * 100
            s.loc['Buy & Hold Return [%]'] = (data.buy_and_hold_equity[-1] - data.buy_and_hold_equity[0]) / data.buy_and_hold_equity[0] * 100
            s.loc['Mean Return Per Day'] = return_per_day = (trade_return+1).prod()**(1/data.shape[0]) - 1
            s.loc['Mean Net Return Per Day'] = net_return_per_day = (data.strategy_net_return+1).prod()**(1/data.shape[0]) - 1
            s.loc['Annualized Return [%]'] = annualized_return = ((net_return_per_day+1)**252 - 1) * 100
            s.loc['Annualized Volatility'] = annualized_volatility = net_trade_return.std()*np.sqrt(252)
            s.loc['# Trades'] = trade_count = len(sell_dates)
            s.loc['# Trades Per Year'] = trade_count_per_year = trade_count / (len(data)/252)
            s.loc['Win Rate [%]'] = win_rate = (net_trade_return > 0).sum() / trade_count * 100
            s.loc['Best Trade [%]'] = data.strategy_net_return.max() * 100
            s.loc['Worst Trade [%]'] = data.strategy_net_return.min() * 100
            dd = 1 - data.strategy_net_return / np.maximum.accumulate(data.strategy_net_return)
            dd_dur, dd_peaks = _compute_drawdown_duration_peaks(pd.Series(dd, index=data.index))
            s.loc['Max. Drawdown Date'] = simple_drawdown(data.strategy_net_return)["drawdown"].idxmin()
            s.loc['Max. Drawdown [%]'] = max_dd = -np.nan_to_num(dd.max()) * 100
            s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
            s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
            s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())
            s.loc['Annualized Sharpe Ratio'] = (annualized_return - 0.01) / annualized_volatility
            s.loc['Calmar Ratio'] = annualized_return / ((-max_dd / 100) or np.nan)
            print(s)

        def plot_drawdown_underwater(returns, ax=None, **kwargs):
            # Reference from https://github.com/quantopian/pyfolio/blob/master/pyfolio/plotting.py
            def percentage(x, pos):
                """
                Adds percentage sign to plot ticks.
                """
                return '%.0f%%' % x

            if ax is None:
                ax = plt.gca()

            y_axis_formatter = FuncFormatter(percentage)
            ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

            df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
            running_max = np.maximum.accumulate(df_cum_rets)
            underwater = -100 * ((running_max - df_cum_rets) / running_max)
            (underwater).plot(ax=ax, kind='area', color='coral', alpha=0.7, **kwargs)
            ax.set_ylabel('Drawdown')
            ax.set_title('Underwater plot')
            ax.set_xlabel('')
            return ax

        # Plot price and signal
        if plot:
            #fig = plt.figure(figsize=(20, 15))
            fig = make_subplots(rows = 7,cols = 1,shared_xaxes = True,
            vertical_spacing=0.02,
            subplot_titles=("Close Price","Equity", "Time Under Water","Signal"))

            #Close Price
            fig.add_trace(go.Scatter((
                x = data.loc[data.buy_price != 0].index, y = data.loc[data.buy_price != 0,"Close"],mode = 'markers',
                marker_symbol = 5,marker_color = "green",name = 'Buy')
            ), row=1, col=1)
            #x is data y = plot

            fig.add_trace(go.Scatter(
                x = data.loc[data.sell_price != 0].index, y = data.loc[data.sell_price != 0,"Close"],mode = 'markers',
                marker_symbol = 6,marker_color = "red",name = 'Sell')
            ), row=1, col=1)
            #Plotted with shared x axes please add desired range of dates as x
            for i in range(len(data)):
                if data.binary_signal[i] == 1:
                    ax1.axvspan(
                        mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) - 0.5,
                        mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) + 0.5,
                        facecolor='lightgreen', edgecolor='none', alpha=0.5
                        )
                else:
                    ax1.axvspan(
                        mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) - 0.5,
                        mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) + 0.5,
                        facecolor='lightcoral', edgecolor='none', alpha=0.5
                        )

            #Equity
            fig.add_trace(go.Scatter(
                x= x,
                y= data.buy_and_hold_equity,
                name = "Buy & Hold"
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x= x,
                y= data.strategy_equity,
                name = "{} Strategy".format(self.strategy)"
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x= x,
                y= data.strategy_net_equity,
                name = "{} Strategy with Fee".format(self.strategy)
            ), row=2, col=1)

            #Time underwater

            fig.add_trace(go.Scatter(
                x= x,
                y= data.strategy_net_return,
                name = "{} Strategy with Fee".format(self.strategy)
            ), row=3, col=1)

            fig.add_trace(go.Scatter(
                x= x,
                y = pd.Series(y_pred, index=self.dtest.index),
                name = "{} Strategy with Fee".format(self.strategy)
            ), row=4, col=1)

            fig.update_layout(
            autosize=False,
            width=800,
            height=500,
            title_x=0.5)
            fig.show()
            #spec = gridspec.GridSpec(nrows=7, ncols=1, figure=fig)
            # ax1 = fig.add_subplot(spec[0:2])
            # ax1.plot(data.Close, alpha=0.7)
            # ax1.scatter(data.loc[data.buy_price != 0].index, data.loc[data.buy_price != 0, "Close"], marker="^", label="buy", color="green")
            # ax1.scatter(data.loc[data.sell_price != 0].index, data.loc[data.sell_price != 0, "Close"], marker="v", label="sell", color="red")
            # for i in range(len(data)):
            #     if data.binary_signal[i] == 1:
            #         ax1.axvspan(
            #             mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) - 0.5,
            #             mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) + 0.5,
            #             facecolor='lightgreen', edgecolor='none', alpha=0.5
            #             )
            #     else:
            #         ax1.axvspan(
            #             mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) - 0.5,
            #             mdates.datestr2num(data.index[i].strftime('%Y-%m-%d')) + 0.5,
            #             facecolor='lightcoral', edgecolor='none', alpha=0.5
            #             )
            # ax1.set_xticklabels([])
            # ax1.set_title("Close Price")
            # ax1.legend(loc="upper left")
            # ax1.grid()
            # ax2 = fig.add_subplot(spec[2:4])
            # ax2.plot(data.buy_and_hold_equity, label="Buy & Hold")
            # ax2.plot(data.strategy_equity, label="{} Strategy".format(self.strategy))
            # ax2.plot(data.strategy_net_equity, label="{} Strategy with Fee".format(self.strategy))
            # ax2.set_xticklabels([])
            # ax2.set_title("Equity")
            # ax2.legend(loc="upper left")
            # ax2.grid()
            # ax3 = fig.add_subplot(spec[4:6])
            # ax3 = plot_drawdown_underwater(data.strategy_net_return, ax=ax3)
            # ax3.set_title("Time Under Water")
            # ax3.set_xticklabels([])
            # ax3.grid()
            # ax4 = fig.add_subplot(spec[6])
            # ax4.plot(pd.Series(y_pred, index=self.dtest.index))
            # ax4.set_title("Signal")
            # ax4.grid()
            # plt.tight_layout()
            # plt.show()


def main():
    global Config
    # Get data and split into train dataset and test dataset
    data = yf.download(Config.SYMBOL)

    # Backtest for machine learning
    bt = MLBacktest(data=data, strategy=Config.STRATEGY, cash=Config.CASH, fee=Config.FEE)
    bt.run()

    # Get trained model
    file_name = "./checkpoints/ML_{}.bin".format(datetime.datetime.today().date())
    clf = joblib.load(file_name)


if __name__ == "__main__":
    main()
