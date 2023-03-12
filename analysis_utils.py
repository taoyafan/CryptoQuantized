from pandas import DataFrame
from base_types import IdxValue, DataElements
from data import Data
from plot import PricePlot
from utils import milliseconds_to_date
from base_types import DataType
import json
import pandas as pd
import numpy as np

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


def read_data(symbol, exp_name, start, end) -> SavedInfo:
    start_str = milliseconds_to_date(start) + ' UTC+8'
    end_str = milliseconds_to_date(end + 1) + ' UTC+8'

    data = Data(symbol, DataType.INTERVAL_1MINUTE, start_str=start_str, end_str=end_str, is_futures=True)
    # print(data.start_time())
    # print(data.end_time())

    base_path = '.\\log\\{}\\{}_start_{}_end_{}_'.format(exp_name, symbol, data.start_time(), data.end_time())
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


def get_combined_data(symbol, exp_name, start, end) -> DataFrame:
    info = read_data(symbol, exp_name, start, end)
    df = info.data.data
    df['TR'] = df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
    top_idx = -1
    bottom_idx = -1
    top_time = info.tops.idx
    top_value = info.tops.value
    bottom_time = info.bottoms.idx
    bottom_value = info.bottoms.value

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
        if top_idx + 1 < len(top_time) and open_time >= int(top_time[top_idx + 1]) + 30*60000:
            top_idx += 1

        if bottom_idx + 1 < len(bottom_time) and open_time >= int(bottom_time[bottom_idx + 1]) + 30*60000:
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