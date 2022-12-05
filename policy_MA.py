from new_policy import Policy
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List, Set
from enum import Enum, auto
import json
import os

from base_types import OrderSide, IdxValue, PolicyToAdaptor, OptState, Recoverable
from adapter import Adaptor
from utils import date_to_milliseconds, milliseconds_to_date

class RingBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = np.zeros(size)
        self.num = 0
        self.idx = 0    # The next position to push
    
    def insert(self, data) -> int:
        out_data = 0
        if self.num < self.size:
            self.buffer[self.idx] = data
            self.idx += 1
            self.num += 1
        else:
            if self.idx == self.size:
                self.idx = 0
            out_data = self.buffer[self.idx]
            self.buffer[self.idx] = data
            self.idx += 1

        return out_data
    
    def get_previous_data(self, previous_step: int) -> int:
        assert(previous_step >= 0 and previous_step < self.size)
        # previous_step: 0 Means current data.
        idx = self.idx - 1 - previous_step
        if idx < 0:
            idx = self.size + idx
        else:
            idx = idx

        return self.buffer[idx]


class MA:
    def __init__(self, level, buffer: RingBuffer):
        assert(level <= buffer.size)
        self.sum = 0
        self.level = level
        self.buffer = buffer

        self.mean = 0
        self.mean_pre_1 = 0
        self.mean_pre_2 = 0

    def update(self, new_data, insert_to_buffer=False):
        # 5 6 7  3 4     new data is 8, last data is 3 
        #     ^  ^
        #    idx |
        #     idx - 4
        last_data = self.buffer.get_previous_data(self.level-1)
        self.sum += new_data - last_data
        
        self.mean_pre_2 = self.mean_pre_1
        self.mean_pre_1 = self.mean
        self.mean = self.sum / min(self.level, self.buffer.num+1)

        if insert_to_buffer:
            self.buffer.insert(new_data)
    
    def valid(self) -> bool:
        return self.buffer.num >= self.level


class MAs:

    def __init__(self, ma_levels: List[int]) -> None:
        self.levels = ma_levels
        self.max_level = max(ma_levels)
        self.buffer = RingBuffer(self.max_level)
        self.mas : Dict[int, MA] = {}
        for level in ma_levels:
            self.mas[level] = (MA(level, self.buffer))
    
    def update(self, data):
        for level in self.mas.keys():
            self.mas[level].update(data)
        
        self.buffer.insert(data)
    
    def get_ma(self, level: int):
        return self.mas[level]

# Break up -> buy, Break down -> short
class PolicyMA(Policy):

    def __init__(self, level_fast, level_slow, log_en: bool=True, analyze_en: bool=True):
        super().__init__(log_en, analyze_en)
        self.mas = MAs([level_fast, level_slow, level_slow * 16])
        self.level_slow = level_slow
        self.level_fast = level_fast

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int) -> None:
        self.mas.update(close)

    def _get_params_buy(self) -> PolicyToAdaptor:
        ma_fast = self.mas.get_ma(self.level_fast)
        ma_slow = self.mas.get_ma(self.level_slow)
        ma_very_slow = self.mas.get_ma(self.level_slow * 16)
        if ma_fast.valid() and ma_slow.valid() and \
           (ma_fast.mean - ma_slow.mean) > 0 and \
           (ma_fast.mean_pre_1 - ma_slow.mean_pre_1) <= 0 and \
           (ma_very_slow.mean - ma_very_slow.mean_pre_2) >= 0:

            return self.ORDER_MARKET
        else:
            return self.DONOT_ORDER
            
    def _get_params_sell(self) -> PolicyToAdaptor:
        ma_fast = self.mas.get_ma(self.level_fast)
        ma_slow = self.mas.get_ma(self.level_slow)
        ma_very_slow = self.mas.get_ma(self.level_slow * 16)
        if ma_fast.valid() and ma_slow.valid() and \
           (ma_fast.mean - ma_slow.mean) < 0 and \
           (ma_fast.mean_pre_1 - ma_slow.mean_pre_1) >= 0 and \
           (ma_very_slow.mean - ma_very_slow.mean_pre_2) < 0:
            
            return self.ORDER_MARKET
        else:
            return self.DONOT_ORDER