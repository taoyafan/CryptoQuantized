from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Set
from enum import Enum, auto
from base_types import IdxValue, PolicyToAdaptor, OptState
from adapter import Adaptor

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
        
        if self.analyze_en:
            self.buy_state = OptState(self.buy_reasons, self.sell_reasons)
            self.sell_state = OptState(self.sell_reasons, self.buy_reasons)

    def try_to_buy(self, adaptor: Adaptor) -> bool:
        # Return whether bought
        params_buy = self._get_params_buy()
        actual_price = adaptor.buy(params_buy)

        is_executed = False
        if actual_price:
            is_executed = True
            time_str = adaptor.get_time_str()
            self._log("\n{}: buy, price = {}, expect = {}".format(time_str, actual_price, params_buy.price))

            # Save analyze info
            if self.analyze_en:
                self.buy_state.add_part(adaptor.get_timestamp(), params_buy.price, actual_price, params_buy.reason)

        return is_executed

    def try_to_sell(self, adaptor: Adaptor) -> bool:
        # Return whether sold
        params_sell = self._get_params_sell()
        actual_price = adaptor.sell(params_sell)

        is_executed = False
        if actual_price:
            is_executed = True
            time_str = adaptor.get_time_str()
            self._log("{}: sell, price = {}, expect = {}".format(time_str, actual_price, params_sell.price))

            # Save analyze info
            if self.analyze_en:
                buy_price = self.buy_state.points_actual_price[-1]
                buy_reason = self.buy_state.last_reason
                earn = (actual_price - buy_price) / buy_price   # Not include swap fee
                self.buy_state.add_left_part(params_sell.reason, earn)
                self.sell_state.add_all(adaptor.get_timestamp(), params_sell.price, actual_price, 
                                        params_sell.reason, buy_reason, earn)
                # self._log('Earn without fee: {}'.format(earn))

        return is_executed

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

    def __init__(self, log_en: bool=True, analyze_en: bool=True):
        super().__init__(log_en, analyze_en)
        self.last_top = float('inf')
        self.fake_top = 0
        self.fake_top_idx = 0
        self.last_bottom = 0
        self.fake_bottom = float('inf')
        self.fake_bottom_idx = 0
        self.nums = 0   # num after found the last top or bottom
        self.direction = self.DOWN

        if self.analyze_en:
            self.tops = IdxValue()
            self.bottoms = IdxValue()

    def update(self, high: float, low: float, timestamp: int):
        self.nums += 1
        
        if self.direction == self.UP:
            # Search top
            if high > self.fake_top:
                # Trend continue to move up, update fake top
                self.fake_top = high
                self.fake_top_idx = timestamp
                self.nums = 0
            else:
                # if low < self.fake_bottom:
                #     self.fake_bottom = low
                #     self.fake_bottom_idx = timestamp
                
                if self.nums >= 2:
                    # Trend reverses
                    self.direction = self.DOWN
                    self.nums = 0
                    self.last_top = self.fake_top
                    self.fake_bottom = low
                    self.fake_bottom_idx = timestamp

                    if self.analyze_en:
                        self.tops.add(self.fake_top_idx, self.last_top)
        else:
            # Search bottom
            if low < self.fake_bottom:
                # Trend continuely to move down, update fake bottom
                self.fake_bottom = low
                self.fake_bottom_idx = timestamp
                self.nums = 0
            else:
                # if high > self.fake_top:
                #     self.fake_top = high
                #     self.fake_top_idx = timestamp
                
                if self.nums >= 2:
                    # Trend reverses
                    self.direction = self.UP
                    self.nums = 0
                    self.last_bottom = self.fake_bottom
                    self.fake_top = high
                    self.fake_top_idx = timestamp
                    
                    if self.analyze_en:
                        self.bottoms.add(self.fake_bottom_idx, self.last_bottom)

    def _get_params_buy(self) -> PolicyToAdaptor:
        return PolicyToAdaptor(self.last_top, PolicyToAdaptor.ABOVE, 'Default')
    
    def _get_params_sell(self) -> PolicyToAdaptor:
        return PolicyToAdaptor(self.last_bottom, PolicyToAdaptor.BELLOW, 'Default')

    @property
    def buy_reasons(self) -> Set[str]:
        return {'Default'}

    @property
    def sell_reasons(self) -> Set[str]:
        return {'Default'}