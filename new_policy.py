from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List, Set
from enum import Enum, auto
import json
import os

from base_types import OrderSide, IdxValue, PolicyToAdaptor, OptState, Recoverable
from adapter import Adaptor
from utils import date_to_milliseconds, milliseconds_to_date

# Base class of policy
class Policy(ABC):

    class PointsType(Enum):
        EXPECT_BUY = auto()
        ACTUAL_BUY = auto()
        EXPECT_SELL = auto()
        ACTUAL_SELL = auto()

    def __init__(self, log_en: bool, analyze_en):
        self.log_en = log_en            # Whether log
        self.analyze_en = analyze_en    # Whether save analyze data
        self.price_last_trade = 0
        
        if self.analyze_en:
            self.buy_state = OptState(self.buy_reasons, self.sell_reasons)
            self.sell_state = OptState(self.sell_reasons, self.buy_reasons)

    def try_to_buy(self, adaptor: Adaptor) -> bool:
        # Return whether bought
        params_buy = self._get_params_buy()
        actual_price = adaptor.buy(params_buy)

        self.update_info(adaptor, OrderSide.BUY, params_buy, actual_price)

        return True if actual_price else False

    def try_to_sell(self, adaptor: Adaptor) -> bool:
        # Return whether sold
        params_sell = self._get_params_sell()
        actual_price = adaptor.sell(params_sell)
        self.update_info(adaptor, OrderSide.SELL, params_sell, actual_price)

        return True if actual_price else False

    def update_info(self, adaptor: Adaptor, side: OrderSide, params: PolicyToAdaptor, actual_price: Optional[float]):
        if actual_price:
            time_str = adaptor.get_time_str()
            if side == OrderSide.BUY:
                loss = (1 - params.price / actual_price)
            else:
                loss = (1 - actual_price / params.price)
            self._log("{}: {}, price = {}, expect = {}, loss = {:.3f}%".format(
                time_str, side.value, actual_price, params.price, loss*100))

            # Save analyze info
            earn_rate = 0
            if self.price_last_trade > 0:
                if side == OrderSide.BUY:
                    # Earn of last sell
                    earn_rate = (self.price_last_trade - actual_price) / self.price_last_trade   # Not include swap fee
                else:
                    earn_rate = (actual_price - self.price_last_trade) / self.price_last_trade   # Not include swap fee

                self._log('Earn rate without fee: {:.3f}%'.format(earn_rate*100))
            
            if self.analyze_en:
                side_state: OptState = self.buy_state if side == OrderSide.BUY else self.sell_state
                other_state: OptState = self.sell_state if side == OrderSide.BUY else self.buy_state
                
                if other_state.pair_unfinished:
                    other_state.add_left_part(params.reason, earn_rate)

                side_state.add_part(adaptor.get_timestamp(), params.price, actual_price, params.reason)

            self.price_last_trade = actual_price

    def get_points(self, points_type: PointsType) -> IdxValue:
        # The idx is timestamp
        if points_type == self.PointsType.EXPECT_BUY:
            return IdxValue(self.buy_state.points_idx, self.buy_state.points_expect_price)
        elif points_type == self.PointsType.ACTUAL_BUY:
            return IdxValue(self.buy_state.points_idx, self.buy_state.points_actual_price)
        elif points_type == self.PointsType.EXPECT_SELL:
            return IdxValue(self.sell_state.points_idx, self.sell_state.points_expect_price)
        elif points_type == self.PointsType.ACTUAL_SELL:
            return IdxValue(self.sell_state.points_idx, self.sell_state.points_actual_price)
        else:
            raise ValueError

    def log_analyzed_info(self):
        if self.analyze_en:
            print('\nPolicy analyze (without fee):')
            self.buy_state.log('buy', 'sell')
            print()
            self.sell_state.log('sell', 'buy')

    @abstractmethod
    def update(self, high: float, low: float, timestamp: int) -> None:
        return
    
    @abstractmethod
    def _get_params_buy(self) -> PolicyToAdaptor:
        return
    
    @abstractmethod
    def _get_params_sell(self) -> PolicyToAdaptor:
        return

    @property
    @abstractmethod
    def buy_reasons(self) -> Set[str]:
        return

    @property
    @abstractmethod
    def sell_reasons(self) -> Set[str]:
        return

    def save(self, file_loc: str, symbol: str, start, end):
        if self.analyze_en:
            trade_info = {
                'buy_time': self.buy_state.points_idx,
                'buy_price': self.buy_state.points_actual_price,
                'sell_time': self.sell_state.points_idx,
                'sell_price': self.sell_state.points_actual_price,
            }

            file_path = os.path.join(file_loc, '{}_start_{}_end_{}_trade_info.json'.format(symbol, start, end))
            with open(file_path, 'w') as f:
                json.dump(trade_info, f, indent=2)

    def _log(self, s=''):
        if self.log_en:
            print(s)

# Buy when break through prior high
# Sell when break through prior low
class PolicyBreakThrough(Policy):

    MIN_THRESHOLD = 30

    def __init__(self, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False, **kwargs):
        super().__init__(log_en, analyze_en)
        self.highs = np.empty(0)
        self.lows = np.empty(0)
        self.finding_bottom = True

        self.k_same_points_delta = kwargs['k_same_points_delta']
        self.k_other_points_delta = kwargs['k_other_points_delta']
        self.k_from_latest_point = kwargs['k_from_latest_point']
        self.search_to_now = kwargs['search_to_now']

        self.front_threshold = 1

        self.last_checked_time = time

        self.last_top = float('inf')
        self.last_top_time: Recoverable = Recoverable(time)
        self.delta_time_top: Recoverable = Recoverable(0)

        self.last_bottom = 0
        self.last_bottom_time: Recoverable = Recoverable(time)
        self.delta_time_bottom: Recoverable = Recoverable(0)

        self.policy_private_log = policy_private_log

        if self.analyze_en:
            self.tops = IdxValue()
            self.bottoms = IdxValue()

    def _update_threshold(self, checked_time):
        # Last point
        time_last_same_point = self.last_bottom_time.value if self.finding_bottom else self.last_top_time.value
        time_last_other_point = self.last_top_time.value if self.finding_bottom else self.last_bottom_time.value
        time_latest_point = max(time_last_same_point, time_last_other_point)
        
        # delta time
        delta_time_same_point = self.delta_time_bottom.value if self.finding_bottom else self.delta_time_top.value
        delta_time_other_point = self.delta_time_top.value if self.finding_bottom else self.delta_time_bottom.value
        delta_time_latest_point = (checked_time - time_latest_point) * self.k_from_latest_point

        # next_point min time
        next_point_time_min = time_last_same_point + self.k_same_points_delta * delta_time_same_point
        next_other_point_time_min = time_last_other_point + self.k_other_points_delta * delta_time_other_point

        th_same_point_delta = int((next_point_time_min - checked_time) // 60000) + self.MIN_THRESHOLD
        th_other_point_delta = int((next_other_point_time_min - checked_time) // 60000) + self.MIN_THRESHOLD
        th_latest_point_delta = int(delta_time_latest_point // 60000)

        # self.threshold = max(self.MIN_THRESHOLD, th_same_point_delta, th_other_point_delta, th_latest_point_delta)
        self.threshold = max(self.MIN_THRESHOLD, th_same_point_delta, th_latest_point_delta)

    def update(self, high: float, low: float, timestamp: int):
        # Update highs, lows
        self.highs = np.append(self.highs, high)
        self.lows = np.append(self.lows, low)

        while True:
            # For each checked time, update threshold, confirmed time
            self._update_threshold(self.last_checked_time + 60000)
            confirmed_time_of_next_time = self.last_checked_time + (self.threshold + 1) * 60000

            # Doesn't come to the confirmed time, then break
            if confirmed_time_of_next_time > timestamp:
                break

            # Then we can check the next time
            self.last_checked_time += 60000
            idx = len(self.highs) - 1 - (timestamp - self.last_checked_time) // 60000

            # If threshold is 1, We must has three data, idx is len - threshold - 1
            # e.g. threshold is 1, len is 3, then the high/low we checked is highs/lows[1]
            if idx >= self.front_threshold:
                assert len(self.highs) == len(self.lows)
                
                found_top = False
                found_bottom = False

                end_idx = len(self.highs) if self.search_to_now else idx+self.threshold+1

                # Finding bottom and top
                if self.finding_bottom:
                    time_after_last = self.last_checked_time - self.last_bottom_time.value
                    min_delta = self.k_other_points_delta * self.delta_time_bottom.value

                    if (self.lows[idx] <= self.lows[idx: end_idx]).all() and (
                        self.lows[idx] <= self.lows[idx-self.front_threshold: idx]).all() and (
                        time_after_last > min_delta):
                        # Is bottom
                        found_bottom = True
                        self.finding_bottom = False

                else:
                    time_after_last = self.last_checked_time - self.last_top_time.value
                    min_delta = self.k_other_points_delta * self.delta_time_top.value

                    if (self.highs[idx] >= self.highs[idx: end_idx]).all() and (
                        self.highs[idx] >= self.highs[idx-self.front_threshold: idx]).all() and (
                        time_after_last > min_delta):
                        # Is top
                        found_top = True
                        self.finding_bottom = True

                # If found
                if found_top or found_bottom:
                    self._update_points(idx, timestamp, found_top, found_bottom)
                # If not found the top and bottom check whether latest point is fake
                else:
                    if self.finding_bottom and self.highs[idx] > self.last_top:
                        # Found new top when searching bottom, last top is fake top, keep search top
                        self.finding_bottom = False
                        # Revert time info of last top
                        self.delta_time_top.recover()
                        self.last_top_time.recover()
                        self.last_checked_time -= 60000
                        
                    elif self.finding_bottom == False and self.lows[idx] < self.last_bottom:
                        # Found new bottom when searching top, last bottom is fake bottom, keep search bottom
                        self.finding_bottom = True
                        # Revert time info of last bottom
                        self.delta_time_bottom.recover()
                        self.last_bottom_time.recover()

            # if idx >= self.front_threshold:
        # while confirmed_time <= timestamp:
        return

    def _get_params_buy(self) -> PolicyToAdaptor:
        idx = self.front_threshold + 1
        fake_top = np.max(self.highs[idx:]) if len(self.highs) > idx else 0
        return PolicyToAdaptor(max(self.last_top, fake_top), PolicyToAdaptor.ABOVE, 'Default')
    
    def _get_params_sell(self) -> PolicyToAdaptor:
        idx = self.front_threshold + 1
        fake_bottom = np.min(self.lows[idx:]) if len(self.lows) > idx else float('inf')
        return PolicyToAdaptor(min(self.last_bottom, fake_bottom), PolicyToAdaptor.BELLOW, 'Default')

    def save(self, file_loc: str, symbol: str, start, end):
        if self.analyze_en:
            vertices = {
                'top_time': self.tops.idx,
                'top_value': self.tops.value,
                'bottom_time': self.bottoms.idx,
                'bottom_value': self.bottoms.value,
            }

            file_path = os.path.join(file_loc, '{}_start_{}_end_{}_vertices.json'.format(symbol, start, end))
            with open(file_path, 'w') as f:
                json.dump(vertices, f, indent=2)
            
            super().save(file_loc, symbol, start, end)

    @property
    def buy_reasons(self) -> Set[str]:
        return {'Default'}

    @property
    def sell_reasons(self) -> Set[str]:
        return {'Default'}

    def _update_points(self, idx, timestamp, found_top, found_bottom):
        if found_top or found_bottom:
            if found_top:
                self.last_top = self.highs[idx]
                self.delta_time_top.set(self.last_checked_time - self.last_top_time.value)
                self.last_top_time.set(self.last_checked_time)
                if self.analyze_en:
                    self.tops.add(self.last_checked_time, self.last_top)
            
            if found_bottom:
                self.last_bottom = self.lows[idx]
                self.delta_time_bottom.set(self.last_checked_time - self.last_bottom_time.value)
                self.last_bottom_time.set(self.last_checked_time)
                if self.analyze_en:
                    self.bottoms.add(self.last_checked_time, self.last_bottom)

            if self.policy_private_log:
                point_type = 'top' if found_top else 'bottom'
                price = self.last_top if found_top else self.last_bottom
                point_time = self.last_top_time.value if found_top else self.last_bottom_time.value

                self._log('{}: Found new {}, \tprice: {:.4f}, \tat  {}'.format(
                    milliseconds_to_date(timestamp), 
                    point_type, 
                    price,
                    milliseconds_to_date(point_time)))

            self.highs = self.highs[idx-self.front_threshold:]
            self.lows = self.lows[idx-self.front_threshold:]

