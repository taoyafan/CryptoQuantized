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
            idx = self.num + idx

        return self.buffer[idx]


class MA:
    def __init__(self, level, buffer: RingBuffer):
        assert(level <= buffer.size)
        self.sum = 0
        self.level = level
        self.buffer = buffer

    def update_before(self, new_data, insert_to_buffer=False):
        last_data = self.buffer.get_previous_data(self.level-2)  # The new data haven't inserted yet
        self.sum += new_data - last_data
        if insert_to_buffer:
            self.buffer.insert(new_data)

    def value(self):
        return self.sum / self.level


class MAs:

    def __init__(self, ma_levels: List) -> None:
        self.num = max(ma_levels)
        self.levels = ma_levels
        self.buffer = np.zeros(self.num)
    
    def update(self, data):
        pass

# Break up -> buy, Break down -> short
class PolicyMA(Policy):

    def __init__(self, step_width: int=30, log_en: bool=True, analyze_en: bool=True):
        super().__init__(log_en, analyze_en)
        self.step_width = step_width
        self.latest_data = np.zeros(step_width)
        self.idx = 0 
        self.num = 0
        self.sum = 0

        self.mean = 0
        self.mean_pre_1 = 0
        self.mean_pre_2 = 0

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int) -> None:
        if self.num < self.step_width:
            self.latest_data[self.idx] = close
            self.idx += 1
            self.sum += close
            self.num += 1
        else:
            if self.idx == self.step_width:
                self.idx = 0
            self.sum -= self.latest_data[self.idx]
            self.sum += close
            self.latest_data[self.idx] = close
            self.idx += 1
        
        self.mean_pre_2 = self.mean_pre_1
        self.mean_pre_1 = self.mean
        self.mean = self.sum / self.step_width

        return

    def _get_params_buy(self) -> PolicyToAdaptor:
        if (self.mean - self.mean_pre_1) > 0 and (self.mean_pre_1 - self.mean_pre_2) <= 0:
            return self.ORDER_MARKET
        else:
            return self.DONOT_ORDER
            
    def _get_params_sell(self) -> PolicyToAdaptor:
        if (self.mean - self.mean_pre_1) < 0 and (self.mean_pre_1 - self.mean_pre_2) >= 0:
            return self.ORDER_MARKET
        else:
            return self.DONOT_ORDER