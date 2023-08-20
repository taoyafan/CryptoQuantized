import time
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List, Set
from enum import Enum, auto
import json
import os

from base_types import OrderSide, IdxValue, TradeInfo, Order, OptState, Recoverable
from adapter import Adaptor
from account_state import AccountState
from utils import date_to_milliseconds, milliseconds_to_date, MAs, RingBuffer
from plot import PricePlot
from data import Data

# Base class of policy
class Policy(ABC):

    class PointsType(Enum):
        EXPECT_BUY = auto()
        ACTUAL_BUY = auto()
        EXPECT_SELL = auto()
        ACTUAL_SELL = auto()

    def __init__(self, state: AccountState, log_en: bool, analyze_en: bool):
        self.account_state: AccountState = state
        self.log_en = log_en            # Whether log
        self.analyze_en = analyze_en    # Whether save analyze data
        self.price_last_trade = 0
        self.buy_order: Optional[Order] = None
        self.sell_order: Optional[Order] = None
        
        if self.analyze_en:
            self.buy_state = OptState(self.buy_reasons, self.sell_reasons)
            self.sell_state = OptState(self.sell_reasons, self.buy_reasons)

    def reset(self):
        return

    def try_to_buy(self, new_step=False) -> bool:
        # Return whether create a new buying order
        params_buy: Optional[Order] = self._get_params_buy(new_step)
        return self._update_order(params_buy)

    def try_to_sell(self, new_step=False) -> bool:
        # Return whether create a new selling order
        params_sell = self._get_params_sell(new_step)
        return self._update_order(params_sell)

    def _update_order(self, new_order: Optional[Order]) -> bool:
        is_created = False
        if new_order:
            side: OrderSide = new_order.side
            new_order.add_traded_call_back(self.update_info)
            new_order_valid = True

            order_exist = self.buy_order if side == OrderSide.BUY else self.sell_order
            order_opposide = self.sell_order if side == OrderSide.BUY else self.buy_order

            # if existed order finished we can just discard it.
            if order_exist and order_exist.is_alive():

                # if new order is equicalent to existed, no need to create a new one
                if order_exist.equivalent_to(new_order):
                    new_order_valid = False
                else:
                    # Otherwise, we need to cancel old order and create a new one
                    is_canceled = self.account_state.cancel_order(order_exist)
                    # Failed to cancel means it is traded.
                    # New params only valid if the cancellation is successful
                    new_order_valid = is_canceled
            
            if new_order_valid:
                # Whether unexited sell order can be canceled
                if (order_opposide                                 and
                    order_opposide.is_alive()                      and
                    order_opposide.has_exit()                      and
                    order_opposide.exit_priority < new_order.enter_priority
                ):
                    new_order.cancel_another_at_state(Order.State.ENTERED, order_opposide)
                    # order_opposide.cancel_another_at_state(Order.State.EXITED, new_order)

                is_created = self.account_state.create_order(new_order)
                if is_created:
                    if side == OrderSide.BUY:
                        self.buy_order = new_order
                    else:
                        self.sell_order = new_order
            else:
                new_order.set_state_to(Order.State.CANCELED)
            # if new_order_valid
        # if new_order

        return is_created

    def update_info(self, trade: TradeInfo):
        actual_price  = trade.executed_price
        executed_time = trade.executed_time
        side          = trade.side
        price         = trade.price
        reason        = trade.reason
        reduce_only   = trade.reduce_only

        if actual_price and executed_time:
            time_str = milliseconds_to_date(executed_time)
            if side == OrderSide.BUY:
                loss = (1 - price / actual_price) if price > 0 else 0
            else:
                loss = (1 - actual_price / price) if price > 0 else 0
            
            if loss == 0:
                price = actual_price
            
            self._log("\n{}: {}, price = {:.2f}, expect = {:.2f}, loss = {:.3f}%, reason = {}".format(
                time_str, side.value, actual_price, price, loss*100, reason))

            # Save analyze info
            previous_pos_amount = self.account_state.pos_amount
            earn_rate = 0
            if self.price_last_trade > 0 and previous_pos_amount != 0:
                if side == OrderSide.BUY:
                    # Earn of last sell
                    earn_rate = (self.price_last_trade - actual_price) / self.price_last_trade   # Not include swap fee
                else:
                    earn_rate = (actual_price - self.price_last_trade) / self.price_last_trade   # Not include swap fee

                self._log('Earn rate without fee: {:.3f}%\n'.format(earn_rate*100))
            
            if self.analyze_en:
                side_state: OptState = self.buy_state if side == OrderSide.BUY else self.sell_state
                other_state: OptState = self.sell_state if side == OrderSide.BUY else self.buy_state
                
                if other_state.has_added_part:
                    other_state.add_left_part(reason, earn_rate)

                side_state.add_part(executed_time, price, actual_price, reason, reduce_only)

            self.price_last_trade = actual_price

            self.account_state.update_pos()
            
            # Update might failed, then update again
            retry_cnt = 0
            while previous_pos_amount == self.account_state.pos_amount and retry_cnt < 3:
                time.sleep(1)
                retry_cnt += 1
                self.account_state.update_pos()
                
            assert previous_pos_amount != self.account_state.pos_amount, "Updata account state failed"

            self.account_state.update_analyzed_info()

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
    
    def get_atr60(self):
        return None

    def log_analyzed_info(self):
        if self.analyze_en:
            print('\nPolicy analyze (without fee):')
            self.buy_state.log('buy', 'sell')
            print()
            self.sell_state.log('sell', 'buy')

    @abstractmethod
    def update(self, high: float, low: float, close: float, volume: float, timestamp: int) -> None:
        return
    
    @abstractmethod
    def _get_params_buy(self, new_step=False) -> Optional[Order]:
        return
    
    @abstractmethod
    def _get_params_sell(self, new_step=False) -> Optional[Order]:
        return

    @property
    def buy_reasons(self) -> Set[str]:
        return {'Default'}

    @property
    def sell_reasons(self) -> Set[str]:
        return {'Default'}

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

    def get_plot_points(self, data: Data) -> List[PricePlot.Points]:
        buy_points = self.get_points(self.PointsType.ACTUAL_BUY)
        sell_points = self.get_points(self.PointsType.ACTUAL_SELL)

        points = [
            PricePlot.Points(idx=buy_points.idx, value=buy_points.value, s=90, c='r', label='buy'),
            PricePlot.Points(idx=sell_points.idx, value=sell_points.value, s=90, c='g', label='sell')
        ]

        for point in points:
            point.idx = data.time_list_to_idx(point.idx)

        return points

    def _log(self, s='', end='\n'):
        if self.log_en:
            print(s, end=end)

# Buy when break through prior high
# Sell when break through prior low
class PolicyBreakThrough(Policy):

    MIN_THRESHOLD = 2

    def __init__(self, state: AccountState, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False, **kwargs):
        super().__init__(state, log_en, analyze_en)
        self.highs = np.empty(0)
        self.lows = np.empty(0)
        self.point_times = np.empty(0)
        self.finding_bottom = True

        self.k_same_points_delta = kwargs['k_same_points_delta']
        self.k_other_points_delta = kwargs['k_other_points_delta']
        self.k_from_latest_point = kwargs['k_from_latest_point']
        self.search_to_now = kwargs['search_to_now']

        self.front_threshold = 1

        self.last_checked_time = time

        self.last_top = float('inf')
        self.ll_top = float('inf')
        self.last_top_time = time
        self.delta_time_top = 0

        self.last_bottom = 0.0
        self.ll_bottom = 0.0
        self.last_bottom_time = time
        self.delta_time_bottom = 0

        self.policy_private_log = policy_private_log

        if self.analyze_en:
            self.tops = IdxValue()
            self.bottoms = IdxValue()
            self.tops_confirm = IdxValue()
            self.bottoms_confirm = IdxValue()

    def _update_threshold(self, checked_time):
        # Last point
        time_last_same_point = self.last_bottom_time if self.finding_bottom else self.last_top_time
        time_last_other_point = self.last_top_time if self.finding_bottom else self.last_bottom_time
        time_latest_point = max(time_last_same_point, time_last_other_point)
        
        # delta time
        delta_time_same_point = self.delta_time_bottom if self.finding_bottom else self.delta_time_top
        delta_time_other_point = self.delta_time_top if self.finding_bottom else self.delta_time_bottom
        delta_time_latest_point = (checked_time - time_latest_point) * self.k_from_latest_point

        # next_point min time
        next_point_time_min = time_last_same_point + self.k_same_points_delta * delta_time_same_point
        next_other_point_time_min = time_last_other_point + self.k_other_points_delta * delta_time_other_point

        th_same_point_delta = int((next_point_time_min - checked_time) // 60000) + self.MIN_THRESHOLD
        th_other_point_delta = int((next_other_point_time_min - checked_time) // 60000) + self.MIN_THRESHOLD
        th_latest_point_delta = int(delta_time_latest_point // 60000)

        # self.threshold = max(self.MIN_THRESHOLD, th_same_point_delta, th_other_point_delta, th_latest_point_delta)
        self.threshold = max(self.MIN_THRESHOLD, th_same_point_delta, th_latest_point_delta)

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        # Update highs, lows
        self.highs = np.append(self.highs, high)
        self.lows = np.append(self.lows, low)
        self.point_times = np.append(self.point_times, timestamp)

        while True:
            # 1). Check whether newest data break last confirmed point
            # If finding bottom but get new top, then select the lowest low as bottom
            if self.finding_bottom and self.highs[-1] > self.last_top:
                i_min = self._get_temp_bottom_idx()
                self.last_checked_time = self.point_times[i_min]
                self._update_points(i_min, timestamp, found_top = False, found_bottom = True)
                
                # We need to check this point for bottom too
                self.last_checked_time -= 60000
                self.finding_bottom = False

            # If finding top but get new bottom, then select the highest high as top
            elif (self.finding_bottom == False) and self.lows[-1] < self.last_bottom:
                i_max = self._get_temp_top_idx()
                self.last_checked_time = self.point_times[i_max]
                self._update_points(i_max, timestamp, found_top = True, found_bottom = False)
                
                # We need to check this point for top too
                self.last_checked_time -= 60000
                self.finding_bottom = True

            # 2). Check data with threshold
            # For each checked time, update threshold, confirmed time
            self._update_threshold(self.last_checked_time + 60000)
            confirmed_time_of_next_time = self.last_checked_time + (self.threshold + 1) * 60000

            # Doesn't come to the confirmed time, then break
            if confirmed_time_of_next_time > timestamp:
                break

            # Then we can check the next time
            self.last_checked_time += 60000
            idx = int(len(self.highs) - 1 - (timestamp - self.last_checked_time) // 60000)

            # If threshold is 1, We must has three data, idx is len - threshold - 1
            # e.g. threshold is 1, len is 3, then the high/low we checked is highs/lows[1]
            if idx >= self.front_threshold:
                assert len(self.highs) == len(self.lows)
                
                found_top = False
                found_bottom = False

                end_idx = len(self.highs) if self.search_to_now else idx+self.threshold+1

                # Finding bottom and top
                if self.finding_bottom:
                    time_after_last = self.last_checked_time - self.last_bottom_time
                    min_delta = self.k_other_points_delta * self.delta_time_bottom

                    if (self.lows[idx] <= self.lows[idx: end_idx]).all() and (
                        self.lows[idx] <= self.lows[idx-self.front_threshold: idx]).all() and (
                        time_after_last > min_delta):
                        # Is bottom
                        found_bottom = True
                        self.finding_bottom = False
                        self._update_points(idx, timestamp, found_top, found_bottom)

                else:
                    time_after_last = self.last_checked_time - self.last_top_time
                    min_delta = self.k_other_points_delta * self.delta_time_top

                    if (self.highs[idx] >= self.highs[idx: end_idx]).all() and (
                        self.highs[idx] >= self.highs[idx-self.front_threshold: idx]).all() and (
                        time_after_last > min_delta):
                        # Is top
                        found_top = True
                        self.finding_bottom = True
                        self._update_points(idx, timestamp, found_top, found_bottom)

            # if idx >= self.front_threshold:
        # while confirmed_time <= timestamp:
        return

    def _get_params_buy(self, new_step=False) -> Order:
        return Order(OrderSide.BUY, self._get_latest_top(), Order.ABOVE, 'Default',  # type: ignore
            self.account_state.get_timestamp())
    
    def _get_params_sell(self, new_step=False) -> Order:
        return Order(OrderSide.SELL, self._get_latest_bottom(), Order.BELLOW, 'Default', # type: ignore
            self.account_state.get_timestamp())
    
    def _get_temp_bottom_idx(self):
        assert self.finding_bottom, "Shoud finding bottom"
        lowest = np.inf
        i_min = len(self.lows) - 1
        for i in range(self.front_threshold, len(self.lows)):
            if self.lows[i] < self.lows[i-1] and self.lows[i] < lowest:
                lowest = self.lows[i]
                i_min = i

        return i_min

    def _get_temp_top_idx(self):
        assert self.finding_bottom == False, "Shoud finding top"
        highest = 0
        i_max = len(self.highs) - 1
        for i in range(self.front_threshold, len(self.highs)):
            if self.highs[i] > self.highs[i-1] and self.highs[i] > highest:
                highest = self.highs[i]
                i_max = i

        return i_max

    def _get_latest_bottom(self):
        if self.finding_bottom:
            i_min = self._get_temp_bottom_idx()
            bottom = self.lows[i_min]
        else:
            bottom = self.last_bottom

        return bottom

    def _get_latest_top(self):
        if self.finding_bottom == False:
            i_max = self._get_temp_top_idx()
            top = self.highs[i_max]
        else:
            top = self.last_top

        return top
    
    def save(self, file_loc: str, symbol: str, start, end):
        if self.analyze_en:
            vertices = {
                'top_time': self.tops.idx,
                'top_value': self.tops.value,
                'bottom_time': self.bottoms.idx,
                'bottom_value': self.bottoms.value,
                'tops_confirm_time': self.tops_confirm.idx,
                'tops_confirm_value': self.tops_confirm.value,
                'bottoms_confirm_time': self.bottoms_confirm.idx,
                'bottoms_confirm_value': self.bottoms_confirm.value,
            }

            file_path = os.path.join(file_loc, '{}_start_{}_end_{}_vertices.json'.format(symbol, start, end)) # type: ignore            
            with open(file_path, 'w') as f:
                json.dump(vertices, f, indent=2)
            
            super().save(file_loc, symbol, start, end)

    def get_plot_points(self, data: Data) -> List[PricePlot.Points]:
        tops = self.tops
        bottoms = self.bottoms
        tops_confirm = self.tops_confirm
        bottoms_confirm = self.bottoms_confirm

        points = [
            PricePlot.Points(idx=tops.idx, value=tops.value, s=30, c='b', label='tops'),
            PricePlot.Points(idx=bottoms.idx, value=bottoms.value, s=30, c='y', label='bottoms'),
            PricePlot.Points(idx=tops_confirm.idx, value=tops_confirm.value, s=10, c='m', label='tops_confirm'),
            PricePlot.Points(idx=bottoms_confirm.idx, value=bottoms_confirm.value, s=10, c='orange', label='bottoms_confirm'),
        ]

        for point in points:
            point.idx = data.time_list_to_idx(point.idx)

        points += super().get_plot_points(data)

        return points

    def _update_points(self, idx, timestamp, found_top, found_bottom):
        if found_top or found_bottom:
            if found_top:
                self.ll_top = self.last_top
                self.last_top: float = self.highs[idx] # type: ignore
                self.delta_time_top = self.last_checked_time - self.last_top_time
                self.last_top_time = self.last_checked_time
                if self.analyze_en:
                    self.tops.add(self.last_checked_time, self.last_top)
                    self.tops_confirm.add(timestamp, self.last_top)
            
            if found_bottom:
                self.ll_bottom = self.last_bottom
                self.last_bottom: float = self.lows[idx] # type: ignore
                self.delta_time_bottom = self.last_checked_time - self.last_bottom_time
                self.last_bottom_time = self.last_checked_time
                if self.analyze_en:
                    self.bottoms.add(self.last_checked_time, self.last_bottom)
                    self.bottoms_confirm.add(timestamp, self.last_bottom)

            if self.policy_private_log:
                point_type = 'top' if found_top else 'bottom'
                price = self.last_top if found_top else self.last_bottom
                point_time = self.last_top_time if found_top else self.last_bottom_time

                self._log('{}: Found new {}, \tprice: {:.4f}, \tat  {}'.format(
                    milliseconds_to_date(timestamp), 
                    point_type, 
                    price,
                    milliseconds_to_date(point_time)))

            self.highs = self.highs[idx-self.front_threshold:]
            self.lows = self.lows[idx-self.front_threshold:]
            self.point_times = self.point_times[idx-self.front_threshold:]


class PolicyDelayAfterBreakThrough(PolicyBreakThrough):

    def __init__(self, state: AccountState, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False, **kwargs):
        super().__init__(state, time, log_en, analyze_en, policy_private_log, **kwargs)
        self.break_up = False
        self.break_down = False
        self.break_up_time = 0
        self.break_down_time = 0
        self.timestamp = 0

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        last_top_time_temp = self.last_top_time
        last_bottom_time_temp = self.last_bottom_time
        
        super().update(high, low, close, volume, timestamp)
        
        # Clear break up / down if top or bottom changed.
        if last_top_time_temp != self.last_top_time:
            self.break_up = False
        if last_bottom_time_temp != self.last_bottom_time:
            self.break_down = False

        if high > self.last_top and (not self.break_up):
            self.break_up = True
            self.break_up_time = timestamp
            self.break_down = False

        if low < self.last_bottom and (not self.break_down):
            self.break_up = False
            self.break_down = True
            self.break_down_time = timestamp

        self.timestamp = timestamp + 60000

    def _get_params_buy(self, new_step=False) -> Optional[Order]:
        time_after_last_bottom: int = self.timestamp - self.last_bottom_time
        # min_delta_time: int = self.delta_time_bottom * (2 if self.finding_bottom else 1)
        # min_delta_time: int = self.delta_time_bottom * 2
        min_delta_time: int = 0
        if self.break_up and time_after_last_bottom >= min_delta_time:
            return Order(OrderSide.BUY, 0, Order.ABOVE, 'Default', 
                self.account_state.get_timestamp())
        else:
            return None
    
    def _get_params_sell(self, new_step=False) -> Optional[Order]:
        time_after_last_top: int = self.timestamp - self.last_bottom_time
        # min_delta_time: int = self.delta_time_top * (1 if self.finding_bottom else 2)
        # min_delta_time: int = self.delta_time_top * 2
        min_delta_time: int = 0
        if self.break_down and time_after_last_top >= min_delta_time:
            return Order(OrderSide.SELL, 0, Order.ABOVE, 'Default', 
                self.account_state.get_timestamp())
        else:
            return None


class PolicySwing(PolicyBreakThrough):

    def __init__(self, state: AccountState, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False, **kwargs):
        super().__init__(state, time, log_en, analyze_en, policy_private_log, **kwargs)
        self.can_buy = False
        self.can_sell = False

        self.new_bottom = False
        self.new_top = False
        
        self.last_time = time
        self.last_close = 0.0
        self.last_open = 0.0
        self.sell_order_valid_until = np.inf
        self.buy_order_valid_until = np.inf
        self.atr = MAs([3, 10, 60])
        self.aer = MAs([3, 11, 21, 60])
        self.aer_abs = MAs([11, 21, 60])
        self.ma = MAs([3, 10, 60])
        self.ma10_buffer = RingBuffer(11)
        self.std10abs = MAs([10, 60])
        self.fee = kwargs['fee']
        self.last_ma300 = 0
        
        if self.analyze_en:
            self.atr60_all = np.empty(0)

    def get_atr60(self):
        if self.analyze_en:
            return self.atr60_all
        else:
            return None

    def reset(self):
        self.sell_order_valid_until = np.inf
        self.buy_order_valid_until = np.inf
        self.sell_order = None
        self.buy_order = None
        
        if self.analyze_en:
            self.buy_state.reset()
            self.sell_state.reset()

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        # Whether cancel last order
        if (timestamp >= self.sell_order_valid_until and 
            self.sell_order and 
            self.sell_order.not_entered()
        ):
            self.account_state.cancel_order(self.sell_order)
            self.sell_order_valid_until = np.inf
        
        tr = high - low + 0.000001
        self.atr.update(tr)
        er = (close - self.last_close) / tr
        self.aer.update(er)
        self.aer_abs.update(er if er > 0 else -er)
        self.ma.update(close)
        self.ma10_buffer.insert(self.ma.get_ma(10).mean)
        self.std10abs.update(abs(close - self.ma.get_ma(10).mean))

        if self.analyze_en:
            self.atr60_all = np.append(self.atr60_all, self.atr.get_ma(60).mean / close)

        last_top_temp    = self.last_top
        last_bottom_temp = self.last_bottom
        self.last_time   = timestamp
        self.last_open   = self.last_close  # Use last open to replace open
        self.last_close  = close

        super().update(high, low, close, volume, timestamp)

        
        if (self.can_buy == True and 
            self.highs[-1] > self._buy_price(self.last_top)  # Can not buy after break up
        ):
            self.can_buy = False
        
        if (self.can_sell == True and 
            self.lows[-1] < self.last_bottom  # Can not sell after break down
        ):
            self.can_sell = False

        # Clear break up / down if top or bottom changed.
        if last_top_temp != self.last_top:
            self.can_buy = True
            self.new_top = True

        if last_bottom_temp != self.last_bottom:
            self.can_sell   = True
            self.new_bottom = True

    def _buy_price(self, top):
        if top % 1 < 0.05:
            buy_price = top // 1 + 0.02
        else:
            buy_price = top // 1 + 1.02
        return buy_price

    # AER < 0, In case buying on the short trend
    def _buy_skip_cond1(self):
        atr10 = self.atr.get_ma(10).mean / self.last_close
        atr60 = self.atr.get_ma(60).mean / self.last_close
        
        def price_atr60(price):
            nonlocal atr60
            return (price / self.last_close - 1) / atr60

        # TR condition
        atr10_atr60 = atr10 / atr60
        atr10_cond = atr10_atr60 > 0.6 and atr10_atr60 < 1.78

        # Get MA condition
        ma60_cond = price_atr60(self.ma.get_ma(60).mean) < 5.5

        # Get ER condition
        er = self.aer.get_latest_data()
        aer_10 = self.aer.get_ma(10).mean
        aer_20 = self.aer.get_ma(20).mean
        aer_abs_10 = self.aer_abs.get_ma(10).mean

        er_cond = er < -0.1 or er > 0.05
        aer10_cond = aer_10 < 0.03
        aer20_cond = aer_20 < 0.085
        aerabs10_cond = aer_abs_10 < 0.57

        # Increase of current + top 
        top = self.last_top
        top_atr60 = price_atr60(top)
        # ll_top_atr60 = price_atr60(self.ll_top)
        step_after_top = (self.last_time - self.last_top_time) // 60000
        
        last_open_atr60 = price_atr60(self.last_open)
        inc_in_2step_atr60 = top_atr60 - last_open_atr60

        _2step_inc_cond = inc_in_2step_atr60 > 0.08
        # top_cond = top_atr60 > 0.5
        # atr60_top_cond = top_atr60 > 1.1
        # top_step_cond = step_after_top > 1 
        # ll_top_cond = top_atr60 > 1.8
        
        # Get bottom by atr60
        step_after_bottom = 0
        if len(self.lows) > self.front_threshold:
            if self.finding_bottom:
                bottom = np.min(self.lows[self.front_threshold:])
                ll_bottom = self.last_bottom
                i_min = np.argmin(self.lows[self.front_threshold:]) + self.front_threshold
                step_after_bottom = len(self.lows) - 1 - i_min
            else:
                bottom = self.last_bottom
                ll_bottom = self.ll_bottom
                step_after_bottom = (self.last_time - self.last_bottom_time) // 60000
            
            bottom_atr60 = price_atr60(bottom)
            # ll_bottom_atr60 = price_atr60(ll_bottom)

            last_bottom_cond = bottom_atr60 > -4
            # ll_bottom_cond = ll_bottom_atr60 > -3
            # bottom_step_cond = step_after_bottom < 6.5
            # top_bottom_step_cond = step_after_bottom > 3
        else:
            last_bottom_cond = False
            # ll_bottom_cond = False
            # top_bottom_step_cond = False

        # Step condition
        cycle_step = step_after_top - step_after_bottom
        if cycle_step <= 5:
            step_cond = True
        else:
            # cycle_step > 5, step_after_top > 5
            btm_step_div_top_step = step_after_bottom / step_after_top
            if btm_step_div_top_step <= 0.2:
                step_cond = True
            else:
                step_cond = False



        # Open, Low
        open_atr60 = price_atr60(self.last_open) # Use last close to replace open
        open_cond = open_atr60 < 0.15 or open_atr60 > 0.25

        # can_skip = (True) 
        can_skip = (er_cond and aer10_cond and aer20_cond and aerabs10_cond and ma60_cond and 
                    last_bottom_cond and _2step_inc_cond and open_cond and atr10_cond and step_cond)

        if can_skip:
            self._log(f"{milliseconds_to_date(self.account_state.get_timestamp())}: -- Cond 1 pass, skip")
        
        return can_skip

    # Filter bad first and then filter good of them
    def _buy_skip_bad_good(self):
        top = self.last_top     # It must be the confirmed top
        price_base = top
        
        # Base params ----------------------------------------------------------------

        # - ATR
        atr60 = self.atr.get_ma(60).mean / price_base
        atr3 = self.atr.get_ma(3).mean / price_base / atr60
        atr10 = self.atr.get_ma(10).mean / price_base / atr60

        def price_atr60(price):
            nonlocal atr60
            return (price / price_base - 1) / atr60

        def d_price_atr60(dprice):
            nonlocal atr60
            return dprice / price_base / atr60

        # - AER
        aer11 = self.aer.get_ma(11).mean
        aerabs11 = self.aer_abs.get_ma(11).mean
        aerabs21 = self.aer_abs.get_ma(21).mean
        aerabs60 = self.aer_abs.get_ma(60).mean

        # - std10abs
        std10abs10 = d_price_atr60(self.std10abs.get_ma(10).mean)
        std10abs60 = d_price_atr60(self.std10abs.get_ma(60).mean)

        # Bottom
        step_after_bottom = 0
        if len(self.lows) > self.front_threshold:
            if self.finding_bottom:
                i_min = self._get_temp_bottom_idx()
                step_after_bottom = len(self.lows) - 1 - i_min
            else:
                step_after_bottom = (self.last_time - self.last_bottom_time) // 60000

        # Get steps
        step_after_top = (self.last_time - self.last_top_time) // 60000
        cycle_step = step_after_top - step_after_bottom
        btm_step_div_top_step = step_after_bottom / step_after_top if step_after_top != 0 else 1

        # Bad condition ----------------------------------------------------------------
        log_bad = False

        def log_b_cond(cond, str):
            nonlocal log_bad
            if cond:
                if log_bad == False:
                    log_bad = True
                    self._log(f"{milliseconds_to_date(self.account_state.get_timestamp())}: -- Skip: ", end='')
                self._log(str, end='\t')
            return cond
        
        # - low aerabs60
        b_aerabs60_cond = log_b_cond(aerabs60 < 0.42, f"b_aerabs60_cond pass: {aerabs60 :.3f}")

        # - low aer11
        b_aer11_cond = log_b_cond(aer11 < 0.03, f"b_aer11_cond pass: {aer11 :.3f}")

        # - ATR60 is too low
        b_atr60_cond = log_b_cond(atr60 < 0.0003, f"b_atr60_cond pass: {atr60 :.5f}")

        # - low range around ma 10, i.e. No large range around ma.
        b_std10abs_cond = log_b_cond(std10abs60 < 0.55 or (std10abs60 < 0.8 and std10abs10 < 0.3), 
                                   f"b_std10abs_cond pass: {std10abs60 :.3f}, {std10abs10 :.3f}")

        # - When cycle step is 1.
        b_step_cond = log_b_cond(cycle_step == 1, f"step_le1_cond pass: {cycle_step}")

        bad_cond = (b_aerabs60_cond or b_aer11_cond or b_atr60_cond or b_std10abs_cond or b_step_cond)
        
        good_cond = False
        if bad_cond:
            self._log()

            # Good condition ----------------------------------------------------------------
            
            # ATR std
            atr20arr = self.atr.get_data_arr(20)
            atrstd20 = d_price_atr60(atr20arr.std())
            
            # - close
            close = price_atr60(self.last_close)
            # - ma10 inc
            ma10_inc10 = d_price_atr60(self.ma10_buffer.get_data(0) - self.ma10_buffer.get_data(-10))
            # - 2 step inc
            inc_in_2step = -price_atr60(self.last_open)
            # - ll_top
            ll_top = price_atr60(self.ll_top)
            # - ma3
            ma3 = price_atr60(self.ma.get_ma(3).mean)

            log_good = False

            def log_g_cond(cond, str):
                nonlocal log_good
                if cond:
                    if log_good == False:
                        log_good = True
                        self._log(f"\t -- Bypass skip: ", end='')
                    self._log(str, end='\t')
                return cond
            
            # - TR change a lot in latest 20 cycle
            g_atrstd20_cond   = log_g_cond((atrstd20 > 0.75) and ((close < -1.8) or (atrstd20 > 1.05)), 
                                    f"g_atrstd20_cond pass: {atrstd20 :.3f}, {atrstd20 :.3f}")
            # - high aerabs60
            g_aerabs60_cond   = log_g_cond((aerabs60 > 0.48) and (ma10_inc10 > -0.95), 
                                    f"g_aerabs60_cond pass: {aerabs60 :.3f}, {ma10_inc10 :.3f}")
            # - 2 step increase
            g_2step_inc_cond  = log_g_cond((inc_in_2step < 0.2) and (btm_step_div_top_step < 0.67), 
                                    f"g_2step_inc_cond pass: {inc_in_2step :.2f}, {btm_step_div_top_step :.3f}")
            # - ll top is high
            g_ll_top_cond     = log_g_cond((ll_top > 2.5) and ((aer11 > -0.01) or (close > -0.5)), 
                                    f"g_ll_top_cond pass: {ll_top :.2f}, {aer11 :.3f}, {close :.2f}")
            # - growing not too fast and top is not high
            g_ma3_cond        = log_g_cond((ma3 > -0.45) and ((cycle_step >= 3) or (close > -0.13)), 
                                    f"g_ma3_cond pass: {ma3 :.3f}, {cycle_step}, {close :.3f}")
            # - Cycle step 
            g_cycle_step_cond = log_g_cond((cycle_step == 1) and ((ma10_inc10 > 1.6) or (atr60 > 0.001)), 
                                    f"g_cycle_step_cond pass: {cycle_step}, {ma10_inc10 :.3f}, {atr60 :.4f}")

            good_cond = (g_atrstd20_cond or g_aerabs60_cond or g_2step_inc_cond or
                         g_ll_top_cond or g_ma3_cond or g_cycle_step_cond)
            
            if good_cond:
                self._log()
        
        can_skip = (bad_cond and (not good_cond))

        return can_skip

    # Too top, don't buy
    def _buy_skip_cond2(self):
        atr60 = self.atr.get_ma(60).mean / self.last_close
        
        def price_atr60(price):
            nonlocal atr60
            return (price / self.last_close - 1) / atr60
        
        # Get AER10 condition
        aer_10 = self.aer.get_ma(10).mean
        aer10_cond = aer_10 > 0.15

        # MA3, MA60
        ma3_atr60 = price_atr60(self.ma.get_ma(3).mean)
        ma60_atr60 = price_atr60(self.ma.get_ma(60).mean)
        ma3_cond = ma3_atr60 < 0
        ma60_cond = ma60_atr60 < -1 or ma60_atr60 > 0

        # Top cond
        top_atr60 = price_atr60(self.last_top)
        ll_top_atr60 = price_atr60(self.ll_top)
        step_after_top = (self.last_time - self.last_top_time) // 60000
        
        top_cond = top_atr60 > 1
        ll_top_cond = ll_top_atr60 < 0

        # Step after bottom
        step_after_bottom = 0
        if len(self.lows) > self.front_threshold:
            if self.finding_bottom:
                bottom = np.min(self.lows[self.front_threshold:])
                ll_bottom = self.last_bottom
                i_min = np.argmin(self.lows[self.front_threshold:]) + self.front_threshold
                step_after_bottom = len(self.lows) - 1 - i_min
            else:
                bottom = self.last_bottom
                ll_bottom = self.ll_bottom
                step_after_bottom = (self.last_time - self.last_bottom_time) // 60000

            bottom_atr60 = price_atr60(bottom)
            ll_bottom_atr60 = price_atr60(ll_bottom)
            ll_l_bottom_atr60 = ll_bottom_atr60 - bottom_atr60
            ll_l_bottom_cond = ll_l_bottom_atr60 < 0.2
        else:
            ll_l_bottom_cond = False

        # Open, high cond
        open_atr60 = price_atr60(self.last_open) # Use last close to replace open
        open_cond = open_atr60 < 0

        high_atr60 = price_atr60(self.highs[-1])
        high_cond = high_atr60 < 0.6

        # Is up
        is_up_cond = self.finding_bottom

        # Cycle step condition
        cycle_step = step_after_top - step_after_bottom
        assert cycle_step >= 0
        cycle_step_cond = cycle_step < 3.5 or cycle_step > 6.5

        can_skip = ((is_up_cond or ma3_cond) and ma60_cond and open_cond and 
                    top_cond and high_cond and 
                    ll_top_cond and cycle_step_cond and ll_l_bottom_cond)
        
        if can_skip:
            self._log(f"{milliseconds_to_date(self.account_state.get_timestamp())}: -- Cond 2 pass, skip")
        
        return can_skip

    def _get_params_buy(self, new_step=False) -> Optional[Order]:
        new_order = None
        # On each step
        if (new_step and
            self.atr.get_ma(60).valid()
        ):
            open_time = self.account_state.get_timestamp()

            # 1. Whether old order need cancel ?
            #    -- order exist, not finished, not entered, exceed the valid time.
            need_cancel_exist = ((self.buy_order is not None) and 
                                 (self.buy_order.not_entered()) and
                                 (open_time > self.buy_order_valid_until))

            # Need to cancel when new top found
            if self.new_top:
                # Each top only order once
                self.new_top = False

                if self.buy_order is not None and self.buy_order.not_entered():
                    need_cancel_exist = True

            
            # 2. Whether create new order ? if enter info of new is same as the order need canceled, then don't cancel it
            if ((self.can_buy == True) and
                # No alive buy order or it is not entered
                (self.buy_order == None or 
                 self.buy_order.is_alive() == False or 
                 self.buy_order.not_entered())
            ):
                buy_price = self._buy_price(self.last_top)

                # sell_price = buy_price * (1 + 5 * atr60) # np.mean([earn_tr10, earn_tr60, earn_er10, earn_bottom, earn_ma3])
                # stop_price = buy_price * (1 - 2 * atr60)
                # p = 0.6
                # possible_loss = buy_price - stop_price
                
                # rw = (sell_price - buy_price) / buy_price - 2 * self.fee
                # rl = possible_loss / buy_price + 2 * self.fee
                # leverage = p / rl - (1 - p) / rw
                # leverage = 0.001 / atr60
                leverage = 1
                leverage = min(5, int(leverage // 1))
                leverage = max(1, leverage)

                # if rw > 0 and rl > 0 and leverage > 0:
                # if (leverage > 0):
                if (leverage > 0 and not self._buy_skip_bad_good()):
                # if (leverage > 0 and not (self._buy_skip_cond1() or self._buy_skip_cond2())):
                    order = None
                    
                    # There is a same bought order flying
                    if (self.buy_order and 
                        self.buy_order.is_alive() and
                        self.buy_order.entered_info.price == buy_price 
                    ):  
                        assert self.buy_order.not_entered(), "The order should not entered"
                        # Same enter info, only change exit info.
                        order = self.buy_order
                        order.clear_exit()
                        need_cancel_exist = False

                    else:
                        # No same flying order, create new
                        order = Order(OrderSide.BUY, buy_price, Order.ABOVE, 'Long', 
                            open_time, leverage=leverage, can_be_sent=True, loss_allowed=0.00003)
                        new_order = order

                    if order:
                        # To make sure not move the order to finished
                        order.add_exit(0, Order.ABOVE, "Long timeout", lock_time=int(1.9*60000))
                        # order.add_exit(self.last_bottom, Order.BELLOW, "Long stop", can_be_sent=True, lock_time=1*60000) # , lock_time=1*60000
                        # order.add_exit(sell_price, Order.ABOVE, "Long exit", can_be_sent=True)
                    

                    self.buy_order_valid_until = open_time + 0.9 * 60000      # current one step

            
            # 3. Cancel old order.
            if need_cancel_exist:
                assert (self.buy_order is not None)
                self.account_state.cancel_order(self.buy_order)

        return new_order
    
    def _get_params_sell(self, new_step=False) -> Optional[Order]:
        return None
        new_order = None

        # On each step
        if (new_step and
            self.atr.get_ma(60).valid()
        ):
            open_time = self.account_state.get_timestamp()

            # 1. Whether old order need cancel ?
            #    -- order exist, not finished, not entered, exceed the valid time.
            need_cancel_exist = ((self.sell_order is not None) and 
                                 (self.sell_order.not_entered()) and
                                 (open_time > self.sell_order_valid_until))

            # Need to cancel when new bottom found
            if self.new_bottom:
                # Each bottom only order once
                self.new_bottom = False

                if self.sell_order is not None and self.sell_order.not_entered():
                    need_cancel_exist = True

            
            # 2. Whether create new order ? if enter info of new is same as the order need canceled, then don't cancel it
            if ((self.can_sell == True) and
                # No alive sell order or it is not entered
                (self.sell_order == None or 
                 self.sell_order.is_alive() == False or 
                 self.sell_order.not_entered())
            ):

                atr60 = self.atr.get_ma(60).mean / self.last_close + 0.000001
                atr10 = self.atr.get_ma(10).mean / self.last_close + 0.000001

                bottom = self.last_bottom
                
                def price_atr60(price):
                    nonlocal atr60
                    return (price / self.last_close - 1) / atr60

                aer_11 = self.aer.get_ma(11).mean

                # sell price
                sell_price = bottom

                leverage = 5
                leverage = min(5, int(leverage // 1))
                leverage = max(1, leverage)

                if (leverage > 0): #
                    order = None
                    
                    # There is a same bought order flying
                    if (self.sell_order and 
                        self.sell_order.is_alive() and
                        self.sell_order.entered_info.price == sell_price 
                    ):  
                        assert self.sell_order.not_entered(), "The order should not entered"
                        # Same enter info, only change exit info.
                        order = self.sell_order
                        order.clear_exit()
                        need_cancel_exist = False

                    else:
                        # No same flying order, create new
                        order = Order(OrderSide.SELL, sell_price, Order.BELLOW, 'Short', 
                            open_time, leverage=leverage, can_be_sent=True, loss_allowed=0)
                        new_order = order

                    if order:
                        # To make sure not move the order to finished
                        order.add_exit(0, Order.ABOVE, "Short timeout", lock_time=int(1.5*60000))
                    

                    self.sell_order_valid_until = open_time + 0.9 * 60000      # current one step

            
            # 3. Cancel old order.
            if need_cancel_exist:
                assert (self.sell_order is not None)
                self.account_state.cancel_order(self.sell_order)

        return new_order

    @property
    def buy_reasons(self) -> Set[str]:
        return {'Long', 'Short exit', 'Short stop', 'Short timeout'}

    @property
    def sell_reasons(self) -> Set[str]:
        return {'Short', 'Long exit', 'Long stop', 'Long timeout'}
