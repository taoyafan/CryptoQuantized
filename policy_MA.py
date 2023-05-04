from new_policy import Policy
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List, Set
from enum import Enum, auto
import json
import os

from base_types import OrderSide, IdxValue, Order, OptState, Recoverable
from adapter import Adaptor
from account_state import AccountState
from utils import date_to_milliseconds, milliseconds_to_date, MAs


# Break up -> buy, Break down -> short
class PolicyMA(Policy):

    def __init__(self, state: AccountState, level_fast, level_slow, log_en: bool=True, analyze_en: bool=True):
        super().__init__(state, log_en, analyze_en)
        self.mas = MAs([level_fast, level_slow, level_slow * 16])
        self.level_slow = level_slow
        self.level_fast = level_fast

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int) -> None:
        self.mas.update(close)

    def _get_params_buy(self) -> Optional[Order]:
        ma_fast = self.mas.get_ma(self.level_fast)
        ma_slow = self.mas.get_ma(self.level_slow)
        ma_very_slow = self.mas.get_ma(self.level_slow * 16)
        if ma_fast.valid() and ma_slow.valid() and \
           (ma_fast.mean - ma_slow.mean) > 0 and \
           (ma_fast.mean_pre_1 - ma_slow.mean_pre_1) <= 0 and \
           (ma_very_slow.mean - ma_very_slow.mean_pre_2) >= 0:

            return Order(OrderSide.BUY, 0, Order.ABOVE, 'Default', 
                self.account_state.get_timestamp())
        else:
            return None
            
    def _get_params_sell(self) -> Optional[Order]:
        ma_fast = self.mas.get_ma(self.level_fast)
        ma_slow = self.mas.get_ma(self.level_slow)
        ma_very_slow = self.mas.get_ma(self.level_slow * 16)
        if ma_fast.valid() and ma_slow.valid() and \
           (ma_fast.mean - ma_slow.mean) < 0 and \
           (ma_fast.mean_pre_1 - ma_slow.mean_pre_1) >= 0 and \
           (ma_very_slow.mean - ma_very_slow.mean_pre_2) < 0:
            
            return Order(OrderSide.SELL, 0, Order.ABOVE, 'Default', 
                self.account_state.get_timestamp())
        else:
            return None