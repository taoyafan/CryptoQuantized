from pandas import DataFrame
from base_types import IdxValue, DataElements
from data import Data
from plot import PricePlot
from utils import milliseconds_to_date
from base_types import DataType
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum, auto
from typing import Dict, Optional, List, Set, TypeVar, Generic, Callable
import os

def drop_col(df: pd.DataFrame, name):
    na = [n for n in name if n in df]
    df.drop(na, axis=1, inplace=True)
    
def plot(df: pd.DataFrame, start, end, points=[], mas=[]):
    plot_df = df.loc[start:end, :]
    fig = PricePlot(plot_df['open'], plot_df['high'], plot_df['low'], plot_df['close'], 
            plot_df['open_time'].map(milliseconds_to_date))
    
    for p in points:
        p.idx -= start
        
    fig.plot(points=points, fig=plt.figure(figsize=(25, 8)), mas=mas)

# 画数据分布图
def histplot(pred, truth=None, xlim=None):
    plt.figure(figsize=(12,5))
    sns.histplot(pred, kde=True, color="blue")
    if truth is not None:
        sns.histplot(truth, kde=True, color="red")
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()

# 画 B 相对于 A 的分布图，会把 A 等量分箱 50 份
def plt_A_B(df_data: pd.DataFrame, idx_A, idx_B, bias=0.0, cut=True, cut_num=50):
    plt.figure(figsize=(18,5))

    if cut:
        df_data[idx_A + '_cut'] = pd.qcut(df_data[idx_A], cut_num, duplicates='drop')
        idx_A = idx_A + '_cut'
        
    (df_data.groupby(idx_A)[idx_B].mean() - bias).plot.bar()
    plt.title(idx_B)
    plt.show()

    if cut:
        df_data.drop(idx_A, axis=1, inplace=True)

def plt_point_A_B(df_data, idx_A, idx_B):
    df_data.plot(kind='scatter', x=idx_A, y=idx_B, figsize=(18,5))

# 画 B 相对于 A 的分箱图，会把 A 等量分箱 50 份
def plt_box_A_B(df_data, idx_A, idx_B, cut=True, cut_num=50):
    plt.figure(figsize=(18,5))

    if cut:
        df_data[idx_A + '_cut'] = pd.qcut(df_data[idx_A], cut_num, duplicates='drop')
        idx_A = idx_A + '_cut'

    sns.boxplot(x=idx_A, y=idx_B, data=df_data)
    sns.stripplot(x=idx_A, y=idx_B, data=df_data)
    plt.title(idx_B)
    plt.show()

    if cut:
        df_data.drop(idx_A, axis=1, inplace=True)


def heatmap(pd_data):
    corrmat = pd_data.corr()
    f, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corrmat, vmax=.8, square=True)

def heatmap_font(pd_data, target, k=-1):
    # k is the number of variables for heatmap
    if k == -1:
        k = pd_data.shape[1]
        
    corrmat = pd_data.corr()
    f, ax = plt.subplots(figsize=(20, 16))
    cols = corrmat.nlargest(k, target)[target].index
    cm = np.corrcoef(pd_data[cols].values.T)
    sns.set(font_scale=1.25) # type: ignore
    hm = sns.heatmap(cm,
                     cbar=True,
                     annot=True,
                     square=True,
                     fmt='.2f',
                     annot_kws={'size': 8}, 
                     yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


class Target:
    def __init__(self, target_step = 300, steps = [1, 3, 5, 10, 30, 100, 300]):
        self.target_step = target_step
        self.steps = steps
    
    def reset_target(self, target_step):
        self.target_step = target_step

    def name(self, step: int) -> str:
        return 'earn_after_{}'.format(step)

    def target_name(self):
        return self.name(self.target_step)
    
    def other_names(self):
        return [self.name(s) for s in self.steps if s != self.target_step]
    
    # Inplace
    def add_earns(self, data: pd.DataFrame):
        for step in self.steps:
            data['close_after_{}'.format(step)] = data['close'].shift(-step)
            data[self.name(step)] = (data['close'].shift(-step) - data['close']) / data['close']
            data.drop('close_after_{}'.format(step), axis=1, inplace=True)
        
        return data

    # Inplace
    def drop_others(self, data: pd.DataFrame):
        data.drop(self.other_names(), axis=1, inplace=True)


class FeatTypes(Enum):
    PRICE     = auto()
    DPRICE    = auto()
    TR        = auto()
    VOLUME    = auto()
    VOL_TR    = auto()
    TRADE_NUM = auto()
    TARGET    = auto()
    OTHERS    = auto()


feat_base = {
    FeatTypes.PRICE:     'last_top', 
    FeatTypes.DPRICE:    'last_top', 
    FeatTypes.TR:        'last_top', 
    FeatTypes.VOLUME:    'quote_assert_volume', 
    FeatTypes.VOL_TR:    'volume/TR', 
    FeatTypes.TRADE_NUM: 'number_of_trades', 
    FeatTypes.TARGET:    'target', 
    FeatTypes.OTHERS:    'others' 
}


class Feat:
    def __init__(self, name: str, ftype: FeatTypes):
        self.name  = name    # 10MA
        self.ftype = ftype  # FeatTypes.PRICE
        self.stded = False

    # Standardize feature in place of the input df_data
    def std(self, df_data: pd.DataFrame):
        # Don't divide itself
        if (self.name != feat_base[self.ftype]) and (feat_base[self.ftype] in df_data.columns) and (not self.stded):
            df_data[self.name] = df_data[self.name].div(df_data[feat_base[self.ftype]], axis=0)
            self.stded = True
    
    def set_type(self, ftype: FeatTypes):
        self.ftype = ftype
        # After set neew type, refresh the stded again
        self.stded = False

    def type(self):
        return self.ftype

    def __str__(self):
        return self.name


class FeatData:
    def __init__(self, data: pd.DataFrame, target: Target):
        self.df = data.copy()
        self.target = target
        self.target.add_earns(self.df)
        self.features: Dict[str, Feat] = {}
        self._init_features()

    def drop_other_targets(self):
        self.drop_features(self.target.other_names())

    def add_feature(self, name, ftype, fun: Callable[[pd.DataFrame], pd.Series]):
        self.features[name] = Feat(name, ftype)
        self.df[name] = fun(self.df)

    def cols_of_features(self, feats: List[FeatTypes]) -> List[str]:
        cols = [col for col in self.df.columns if self.features[col].type() in feats]
        return cols

    # One type for all names
    def add_features(self, names, ftypes, fun: Callable[[pd.DataFrame], pd.DataFrame]):
        assert(len(names) == len(ftypes))
        for i in range(len(names)):
            self.features[names[i]] = Feat(names[i], ftypes[i])
        self.df[names] = fun(self.df)

    def add_ave_feature(self, new_name, base_name, ftype, cycles):
        for cc in cycles:
            name = new_name + str(cc)
            self.add_feature(name, ftype, lambda df: df[base_name].rolling(cc).mean())
            
            # k_name = name + '_k'
            # self.add_feature(k_name, ftype, lambda df: df[name] - df[name].shift(1))

    # Features should have one type
    def set_features_type(self, names: List[str], ftype: FeatTypes):
        for na in names:
            self.features[na].set_type(ftype)

    def drop_features(self, names: List[str]):
        self.df.drop(names, axis=1, inplace=True)
        for na in names :
            del self.features[na] 

    def std(self):
        for name in self.features:
            self.features[name].std(self.df)
        
        base_features = set([feat_base[x] for x in list(FeatTypes) if feat_base[x] in self.df])
        self.drop_features(list(base_features))

    def drop_na(self):
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(axis=0, how='any', inplace=True)

    def price_cols(self):
        return self.features

    def _init_features(self):
        exist_feats = [ 
            [['open', 'high', 'low', 'close', 'last_top', 'last_bottom', 'buy_price', 'sell_price'], FeatTypes.PRICE],
            [['TR'], FeatTypes.TR],
            [['number_of_trades'], FeatTypes.TRADE_NUM],
            [['quote_assert_volume', 'taker_buy_quote_asset_volume'], FeatTypes.VOLUME],
            [[self.target.target_name()] + self.target.other_names(), FeatTypes.TARGET]
            ]

        for feats_type in exist_feats:
            feats, ftype = feats_type
            for f in feats:
                if f in self.df.columns:
                    self.features[f] = Feat(f, ftype)

        for col in self.df.columns:
            if col not in self.features.keys():
                self.features[col] = Feat(col, FeatTypes.OTHERS)


def add_ave_features(data: FeatData, cycles = [3, 5, 10, 30, 100, 300], feat_names=['close', 'TR'], new_ave_of_base=None):
    if new_ave_of_base is None:
        new_ave_of_base = {
            'MA': ['close', FeatTypes.PRICE],
            'ATR': ['TR', FeatTypes.TR],
            'AQAV': ['quote_assert_volume', FeatTypes.VOLUME],
            'AT': ['number_of_trades', FeatTypes.TRADE_NUM],
            'ATBQAV': ['taker_buy_quote_asset_volume', FeatTypes.VOLUME],
            'ACS': ['cycle_step', FeatTypes.OTHERS],
            'AVT': ['volume/TR', FeatTypes.VOL_TR],
            }

    for new_name in new_ave_of_base:  
        [base, ftype] = new_ave_of_base[new_name]
        if base in feat_names:
            data.add_ave_feature(new_name, base, ftype, cycles)


def add_macd(data: FeatData, base='close', ftype=FeatTypes.PRICE, fast_period=12, slow_period=26, signal_period=9):
    """
    计算 MACD

    :param df: pandas.DataFrame 包含价格数据
    :param fast_period: int 快线周期, 默认为12
    :param slow_period: int 慢线周期, 默认为26
    :param signal_period: int 信号线周期, 默认为9
    :return: pandas.DataFrame 包含 MACD、Signal 和 Histogram
    """

    # 计算快线、慢线和差离值
    data.add_feature('EMA_fast_'+base, ftype, lambda df: df[base].ewm(span=fast_period).mean())
    data.add_feature('EMA_slow_'+base, ftype, lambda df: df[base].ewm(span=slow_period).mean())
    data.add_feature('MACD_'+base, ftype, lambda df: df['EMA_fast_'+base] - df['EMA_slow_'+base])

    # 计算信号线和 MACD 柱状图
    data.add_feature('signal_'+base, ftype, lambda df: df['MACD_'+base].ewm(span=signal_period).mean())
    data.add_feature('histogram_'+base, ftype, lambda df: df['MACD_'+base] - df['signal_'+base])

    data.drop_features(['EMA_fast_'+base, 'EMA_slow_'+base])


# 显示数据缺失率，最大类别占比
def show_data_basic_info(pd_data, sort_target = 'Percentage of missing values'):
    stats = []
    for col in pd_data.columns:
        stats.append((col,
                      pd_data[col].nunique(),
                      pd_data[col].isnull().sum(),
                      pd_data[col].isnull().sum() * 100 / pd_data.shape[0],
                      pd_data[col].value_counts(normalize=True, dropna=False).values[0] * 100,
                      pd_data[col].dtype))

    stats_df = pd.DataFrame(stats, columns=['Feat',
                                            'Unique_values',
                                            'missing number',
                                            'Percentage of missing values',
                                            'Percentage of values in the biggest category',
                                            'type'])
    
    return stats_df.sort_values(sort_target, ascending=False)


class SavedInfo():
    def __init__(self, data: Data, buy_points: IdxValue, sell_points: IdxValue, 
                 tops: IdxValue, bottoms: IdxValue, earn_points: IdxValue, 
                 tops_confirm: IdxValue, bottoms_confirm: IdxValue):
        self.data: Data = data
        self.buy_points: IdxValue = buy_points    # time, value
        self.sell_points: IdxValue = sell_points  # time, value
        self.tops: IdxValue = tops                # time, value
        self.bottoms: IdxValue = bottoms          # time, value
        self.earn_points: IdxValue = earn_points
        self.tops_confirm: IdxValue = tops_confirm
        self.bottoms_confirm: IdxValue = bottoms_confirm
    
    def get_all(self):
        return self.data, self.buy_points, self.sell_points, self.tops, self.bottoms, self.earn_points, \
               self.tops_confirm, self.bottoms_confirm


def read_data(symbol, exp_name, start, end, is_futures) -> SavedInfo:
    start_str = milliseconds_to_date(start) + ' UTC+8'
    end_str = milliseconds_to_date(end + 1) + ' UTC+8'

    data = Data(symbol, DataType.INTERVAL_1MINUTE, start_str=start_str, end_str=end_str, is_futures=is_futures)
    # print(data.start_time())
    # print(data.end_time())

    path = os.getcwd()
    file_path = os.path.join(path,'log')
    file_path = os.path.join(file_path,'{}'.format(exp_name))
    base_path = os.path.join(file_path, '{}_start_{}_end_{}_'.format(symbol, data.start_time(), data.end_time()))
    trade_info_path = base_path + 'trade_info.json'
    vertices_path = base_path + 'vertices.json'
    earn_path = base_path + 'earn_points.json'

    with open(trade_info_path, 'r') as f:
        json_data = f.read()
        trade_info = json.loads(json_data)

    with open(vertices_path, 'r') as f:
        json_data = f.read()
        vertices = json.loads(json_data)

    with open(earn_path, 'r') as f:
        json_data = f.read()
        earn_points_dict = json.loads(json_data)

    # print(trade_info.keys())
    # print(vertices.keys())
    # print(earn_points_dict.keys())

    buy_points = IdxValue(trade_info['buy_time'], trade_info['buy_price'])
    sell_points = IdxValue(trade_info['sell_time'], trade_info['sell_price'])
    tops = IdxValue(vertices['top_time'], vertices['top_value'])
    bottoms = IdxValue(vertices['bottom_time'], vertices['bottom_value'])
    tops_confirm = IdxValue(vertices['tops_confirm_time'], vertices['tops_confirm_value'])
    bottoms_confirm = IdxValue(vertices['bottoms_confirm_time'], vertices['bottoms_confirm_value'])
    earn_points = IdxValue(earn_points_dict['earn_idx'], earn_points_dict['earn_value'])
    
    return SavedInfo(data, buy_points, sell_points, tops, bottoms, earn_points, tops_confirm, bottoms_confirm)


def get_combined_data(symbol, exp_name, start, end, is_futures, confirm_step=30) -> DataFrame:
    info = read_data(symbol, exp_name, start, end, is_futures)
    df = info.data.data
    df['TR'] = df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
    top_idx = -1
    bottom_idx = -1
    top_time = info.tops.idx
    confirm_top_time = info.tops_confirm.idx
    top_value = info.tops_confirm.value
    
    bottom_time = info.bottoms.idx
    confirm_bottom_time = info.bottoms_confirm.idx
    bottom_value = info.bottoms_confirm.value

    # earn = target_sell_price - current_price
    # target_sell_price = the first sell_price which is not NaN after now
    # sell_price = Nan if low > last_bottom else min(high, last_bottom)
    # buy_price = Nan if high < last_top else max(low, last_top)

    new_col = ['last_top', 'step_after_top', 'last_bottom', 'step_after_bottom','is_up', 'cycle_step', 'buy_price', 'sell_price']
    df = pd.concat([df, 
        pd.DataFrame(columns=new_col)], sort=False)

    def fun(x):
        nonlocal top_idx, bottom_idx
        open_time = x['open_time']

        # Update top_idx and bottom_idx
        if top_idx + 1 < len(confirm_top_time) and open_time >= int(confirm_top_time[top_idx + 1]):
            top_idx += 1

        if bottom_idx + 1 < len(confirm_bottom_time) and open_time >= int(confirm_bottom_time[bottom_idx + 1]):
            bottom_idx += 1
        
        # Get last top and bottom
        if top_idx >= 0:
            last_top = top_value[top_idx]
            step_after_top = (open_time - top_time[top_idx]) / 60000
        else:
            last_top = np.nan
            step_after_top = np.nan
            
        if bottom_idx >= 0:
            last_bottom = bottom_value[bottom_idx]
            step_after_bottom = (open_time - bottom_time[bottom_idx]) / 60000
        else:
            last_bottom = np.nan
            step_after_bottom = np.nan
        
        # Get is_up and cycle_step
        is_up = 1 if step_after_top > step_after_bottom else 0
        cycle_step = step_after_top - step_after_bottom
        cycle_step = cycle_step if cycle_step >= 0 else -cycle_step

        buy_price = max(x['low'], last_top) if x['high'] >= last_top else np.nan
        sell_price = min(x['high'], last_bottom) if x['low'] <= last_bottom else np.nan

        return last_top, step_after_top, last_bottom, step_after_bottom, is_up, cycle_step, buy_price, sell_price

    df[new_col] = df.apply(
        lambda x: fun(x), axis=1, result_type="expand")  # type: ignore
    
    return df