from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Set
from enum import Enum, auto
import json
import os

from base_types import OrderSide, IdxValue, PolicyToAdaptor, OptState
from adapter import Adaptor
from utils import milliseconds_to_date

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
    def update(self, high: float, low: float, open: float, close: float, timestamp: int) -> None:
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

    class Direction(Enum):
        UP = auto()     # Finding the top point 
        DOWN = auto()   # Finding the bottom point
    UP: Direction = Direction.UP
    DOWN = Direction.DOWN

    MIN_THRESHOLD = 20

    def __init__(self, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False):
        super().__init__(log_en, analyze_en)
        self.last_top = float('inf')
        self.last_top_idx = time
        self.fake_top = 0
        self.fake_top_idx = time
        self.last_bottom = 0
        self.last_bottom_idx = 0
        self.fake_bottom = float('inf')
        self.fake_bottom_idx = 0
        self.nums_after_top = 0   # num after found the last top
        self.nums_after_bottom = 0   # num after found the last bottom
        self.direction = self.DOWN
        self.threshold = self.MIN_THRESHOLD  # Trend invert threshold number

        self.policy_private_log = policy_private_log

        if self.analyze_en:
            self.tops = IdxValue()
            self.bottoms = IdxValue()

    def update(self, high: float, low: float, open: float, close: float, timestamp: int):
        self.nums_after_top += 1
        self.nums_after_bottom += 1
        
        if self.direction == self.UP:
            # Search top
            if high > self.fake_top:
                # Trend continue to move up, update fake top
                self.fake_top = high
                self.fake_top_idx = timestamp
                self.nums_after_top = 0
                self.threshold = max((timestamp - self.last_bottom_idx) * 0.5 // 60000, self.MIN_THRESHOLD)
                
                if close < open:
                    self.fake_bottom = low
                    self.nums_after_bottom = 0
                else:
                    self.nums_after_bottom = -1
                    self.fake_bottom = float('inf')
                self.fake_bottom_idx = timestamp
            else:
                if low < self.fake_bottom:
                    self.fake_bottom = low
                    self.fake_bottom_idx = timestamp
                    self.nums_after_bottom = 0
                
                if self.nums_after_top >= self.threshold:
                    # Trend reverses
                    if self.policy_private_log:
                        self._log('{}: Found new top, \tprice: {:.4f}, \tat  {}'.format(
                            milliseconds_to_date(timestamp), self.fake_top, 
                            milliseconds_to_date(timestamp - 60000 * self.nums_after_top)))

                    if self.analyze_en:
                        self.tops.add(self.fake_top_idx, self.fake_top)

                    self.direction = self.DOWN
                    self.nums_after_top = 0
                    self.last_top = self.fake_top
                    self.last_top_idx = self.fake_top_idx
                    self.fake_top = 0
        else:
            # Search bottom
            if low < self.fake_bottom:
                # Trend continuely to move down, update fake bottom
                self.fake_bottom = low
                self.fake_bottom_idx = timestamp
                self.nums_after_bottom = 0
                self.threshold = max((timestamp - self.last_top_idx) * 0.5 // 60000, self.MIN_THRESHOLD)

                if close > open:
                    self.fake_top = high
                    self.nums_after_top = 0
                else:
                    self.fake_top = 0
                    self.nums_after_top = -1
                self.fake_top_idx = timestamp

            else:
                if high > self.fake_top:
                    self.fake_top = high
                    self.fake_top_idx = timestamp
                    self.nums_after_top = 0
                
                if self.nums_after_bottom >= self.threshold:
                    # Trend reverses
                    
                    if self.policy_private_log:
                        self._log('{}: Found new bottom, \tprice: {:.4f} \tat  {}'.format(
                            milliseconds_to_date(timestamp), self.fake_bottom,
                            milliseconds_to_date(timestamp - 60000 * self.nums_after_bottom)))

                    if self.analyze_en:
                        self.bottoms.add(self.fake_bottom_idx, self.fake_bottom)

                    self.direction = self.UP
                    self.nums_after_bottom = 0
                    self.last_bottom = self.fake_bottom
                    self.last_bottom_idx = self.fake_bottom_idx
                    self.fake_bottom = float('inf')

    def _get_params_buy(self) -> PolicyToAdaptor:
        return PolicyToAdaptor(max(self.last_top, self.fake_top), PolicyToAdaptor.ABOVE, 'Default')
    
    def _get_params_sell(self) -> PolicyToAdaptor:
        return PolicyToAdaptor(min(self.last_bottom, self.fake_bottom), PolicyToAdaptor.BELLOW, 'Default')

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