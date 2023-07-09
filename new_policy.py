from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List, Set
from enum import Enum, auto
import json
import os

from base_types import OrderSide, IdxValue, TradeInfo, Order, OptState, Recoverable
from adapter import Adaptor
from account_state import AccountState
from utils import date_to_milliseconds, milliseconds_to_date, MAs
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
            earn_rate = 0
            if self.price_last_trade > 0 and self.account_state.pos_amount != 0:
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

    def _log(self, s=''):
        if self.log_en:
            print(s)

# Buy when break through prior high
# Sell when break through prior low
class PolicyBreakThrough(Policy):

    MIN_THRESHOLD = 3

    def __init__(self, state: AccountState, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False, **kwargs):
        super().__init__(state, log_en, analyze_en)
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

        self.last_bottom = 0.0
        self.last_bottom_time: Recoverable = Recoverable(time)
        self.delta_time_bottom: Recoverable = Recoverable(0)

        self.policy_private_log = policy_private_log

        if self.analyze_en:
            self.tops = IdxValue()
            self.bottoms = IdxValue()
            self.tops_confirm = IdxValue()
            self.bottoms_confirm = IdxValue()

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

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        # Update highs, lows
        self.highs = np.append(self.highs, high)
        self.lows = np.append(self.lows, low)
        recovered = False   # In case two continuously recover.

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
                        self._update_points(idx, timestamp, found_top, found_bottom)

                    # But get new top, then select the lowest low as bottom
                    elif self.highs[idx] > self.last_top and recovered == False:
                        found_bottom = True
                        self.finding_bottom = False

                        i_min = int(np.argmin(self.lows[self.front_threshold: idx])) + self.front_threshold
                        self.last_checked_time = self.last_checked_time - 60000 * (idx - i_min)
                        self._update_points(i_min, timestamp, found_top, found_bottom)


                else:
                    time_after_last = self.last_checked_time - self.last_top_time.value
                    min_delta = self.k_other_points_delta * self.delta_time_top.value

                    if (self.highs[idx] >= self.highs[idx: end_idx]).all() and (
                        self.highs[idx] >= self.highs[idx-self.front_threshold: idx]).all() and (
                        time_after_last > min_delta):
                        # Is top
                        found_top = True
                        self.finding_bottom = True
                        self._update_points(idx, timestamp, found_top, found_bottom)

                    # But get new bottom, then select the highest high as bottom
                    elif self.lows[idx] < self.last_bottom:
                        found_top = True
                        self.finding_bottom = True

                        i_max = int(np.argmax(self.highs[self.front_threshold: idx])) + self.front_threshold
                        self.last_checked_time = self.last_checked_time - 60000 * (idx - i_max)
                        self._update_points(i_max, timestamp, found_top, found_bottom)

            # if idx >= self.front_threshold:
        # while confirmed_time <= timestamp:
        return

    def _get_params_buy(self, new_step=False) -> Order:
        return Order(OrderSide.BUY, self._get_latest_top(), Order.ABOVE, 'Default',  # type: ignore
            self.account_state.get_timestamp())
    
    def _get_params_sell(self, new_step=False) -> Order:
        return Order(OrderSide.SELL, self._get_latest_bottom(), Order.BELLOW, 'Default', # type: ignore
            self.account_state.get_timestamp())

    def _get_latest_top(self):
        idx = self.front_threshold + 1
        fake_top: float = np.max(self.highs[idx:]) if len(self.highs) > idx else 0 # type: ignore
        return max(self.last_top, fake_top)
    
    def _get_latest_bottom(self):
        idx = self.front_threshold + 1
        fake_bottom: float = np.min(self.lows[idx:]) if len(self.lows) > idx else float('inf') # type: ignore
        return min(self.last_bottom, fake_bottom)

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
                self.last_top: float = self.highs[idx] # type: ignore
                self.delta_time_top.set(self.last_checked_time - self.last_top_time.value)
                self.last_top_time.set(self.last_checked_time)
                if self.analyze_en:
                    self.tops.add(self.last_checked_time, self.last_top)
                    self.tops_confirm.add(timestamp, self.last_top)
            
            if found_bottom:
                self.last_bottom: float = self.lows[idx] # type: ignore
                self.delta_time_bottom.set(self.last_checked_time - self.last_bottom_time.value)
                self.last_bottom_time.set(self.last_checked_time)
                if self.analyze_en:
                    self.bottoms.add(self.last_checked_time, self.last_bottom)
                    self.bottoms_confirm.add(timestamp, self.last_bottom)

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


class PolicyDelayAfterBreakThrough(PolicyBreakThrough):

    def __init__(self, state: AccountState, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False, **kwargs):
        super().__init__(state, time, log_en, analyze_en, policy_private_log, **kwargs)
        self.break_up = False
        self.break_down = False
        self.break_up_time = 0
        self.break_down_time = 0
        self.timestamp = 0

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        last_top_time_temp = self.last_top_time.value
        last_bottom_time_temp = self.last_bottom_time.value
        
        super().update(high, low, close, volume, timestamp)
        
        # Clear break up / down if top or bottom changed.
        if last_top_time_temp != self.last_top_time.value:
            self.break_up = False
        if last_bottom_time_temp != self.last_bottom_time.value:
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
        time_after_last_bottom: int = self.timestamp - self.last_bottom_time.value
        # min_delta_time: int = self.delta_time_bottom.value * (2 if self.finding_bottom else 1)
        # min_delta_time: int = self.delta_time_bottom.value * 2
        min_delta_time: int = 0
        if self.break_up and time_after_last_bottom >= min_delta_time:
            return Order(OrderSide.BUY, 0, Order.ABOVE, 'Default', 
                self.account_state.get_timestamp())
        else:
            return None
    
    def _get_params_sell(self, new_step=False) -> Optional[Order]:
        time_after_last_top: int = self.timestamp - self.last_bottom_time.value
        # min_delta_time: int = self.delta_time_top.value * (1 if self.finding_bottom else 2)
        # min_delta_time: int = self.delta_time_top.value * 2
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
        self.bottom_ordered = True
        self.new_top = False
        self.last_time = time
        self.last_close = 0.0
        self.sell_order_valid_until = np.inf
        self.buy_order_valid_until = np.inf
        self.atr = MAs([10, 60])
        self.aer = MAs([10, 60])
        self.ma = MAs([3, 10, 60])
        self.fee = kwargs['fee']
        self.last_ma300 = 0


    def reset(self):
        self.sell_order_valid_until = np.inf
        self.buy_order_valid_until = np.inf
        self.sell_order = None
        self.buy_order = None
        
        if self.analyze_en:
            self.buy_state.reset()
            self.sell_state.reset()

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        self.atr.update(high - low)
        self.aer.update(close - self.last_close)
        self.ma.update(close)

        last_top_temp    = self.last_top
        last_bottom_temp = self.last_bottom
        self.last_time   = timestamp
        self.last_close  = close

        super().update(high, low, close, volume, timestamp)

        # Whether cancel last order
        if (timestamp >= self.sell_order_valid_until and 
            self.sell_order and 
            self.sell_order.not_entered()
        ):
            self.account_state.cancel_order(self.sell_order)
            self.sell_order_valid_until = np.inf
        
        if (timestamp + 59000 > self.buy_order_valid_until and 
            self.buy_order and 
            self.buy_order.not_entered()
        ):
            self.account_state.cancel_order(self.buy_order)
            self.buy_order_valid_until = np.inf
        
        if (self.can_buy == True and 
            self.highs[-1] > self.last_top  # Can not buy after break up
        ):
            self.can_buy = False
        
        # Clear break up / down if top or bottom changed.
        if last_top_temp != self.last_top:
            self.can_buy = True
            self.new_top = True
        if last_bottom_temp != self.last_bottom:
            self.bottom_ordered = False

    def _get_params_buy(self, new_step=False) -> Optional[Order]:
        new_order = None

        if (new_step and
            self.atr.get_ma(60).valid() and
            self.can_buy == True and
            # No alive buy order or it is not entered
            (self.buy_order == None or 
             self.buy_order.is_alive() == False or 
             self.buy_order.not_entered())
        ):

            atr60 = self.atr.get_ma(60).mean / self.last_close + 0.000001
            atr10 = self.atr.get_ma(10).mean / self.last_close + 0.000001
            # aer10 = self.aer.get_ma(10).mean / self.last_close + 0.000001
            # aer60 = self.aer.get_ma(60).mean / self.last_close + 0.000001
            
            ma3 = (self.ma.get_ma(3).mean / self.last_close - 1) / atr60
            ma10 = (self.ma.get_ma(10).mean / self.last_close - 1) / atr60
            ma60 = (self.ma.get_ma(60).mean / self.last_close - 1) / atr60

            top = self._get_latest_top()
            bottom = (self._get_latest_bottom() / self.last_close - 1) / atr60

            buy_price = top + 0.1
            
            # earn_tr10   = 0.5 * atr10
            # earn_tr60   = 1 * atr60
            # earn_er10     = 4 * aer10
            # earn_er60     = 10 * aer60
            # earn_bottom = 0.2 * (top - bottom) / top
            # earn_ma3    = - 1 * ma3 * atr10
            # earn_ma10   = - 1 * ma10 * atr10
            # earn_ma60   = - 0.1 * ma60 * atr10

            # sell_price = buy_price * (1 + 5 * atr60) # np.mean([earn_tr10, earn_tr60, earn_er10, earn_bottom, earn_ma3])
            # stop_price = buy_price * (1 - 2 * atr60)
            # p = 0.6
            # possible_loss = buy_price - stop_price
            
            # rw = (sell_price - buy_price) / buy_price - 2 * self.fee
            # rl = possible_loss / buy_price + 2 * self.fee
            # leverage = p / rl - (1 - p) / rw
            # leverage = min(5, int(leverage // 1))
            leverage = 1
            
            # loss_cond1 = (ma3 > -0.3) and (ma3 < -0.2)
            # loss_cond2 = (ma3 > 0.02) and (ma3 < 0.1)
            # loss_cond3 = (ma3 > 0.4)
            # loss_ma3_cond = loss_cond1 or loss_cond2 or loss_cond3

            # loss_cond4 = (ma10 > 0.3)
            # loss_cond5 = (ma10 > -0.06) and (ma10 < 0.05)
            # loss_ma10_cond = loss_cond4 or loss_cond5
            
            # loss_bottom_cond = bottom > -0.6

            # is_skip = (loss_ma3_cond or loss_ma10_cond) and loss_bottom_cond

            # if rw > 0 and rl > 0 and leverage > 0: # and (not is_skip)
            if leverage > 0:
                open_time = self.account_state.get_timestamp()
                order = None
                
                # There is a same bought order flying
                if (self.buy_order and 
                    self.buy_order.is_alive() and
                    self.buy_order.entered_info.price == buy_price 
                ):  
                    if self.buy_order.not_entered():
                        # Same enter info, only change exit info.
                        order = self.buy_order
                        order.clear_exit()

                else:
                    # No same flying order, create new
                    order = Order(OrderSide.BUY, buy_price, Order.ABOVE, 'Long', 
                        open_time, leverage=leverage, can_be_sent=True)
                    new_order = order

                if order:
                    # To make sure not move the order to finished
                    order.add_exit(0, Order.ABOVE, "Long timeout", lock_time=2*60000)
                    # order.add_exit(stop_price, Order.BELLOW, "Long stop", can_be_sent=True, lock_time=1*60000) # , lock_time=1*60000
                    # order.add_exit(sell_price, Order.ABOVE, "Long exit", can_be_sent=True)
                

                open_time = self.account_state.get_timestamp()
                self.buy_order_valid_until = open_time + 0.5 * 60000      # current one step

        if self.new_top:
            # Each top only order once
            self.new_top = False

            if self.buy_order is not None and self.buy_order.not_entered():
                self.account_state.cancel_order(self.buy_order)

        return new_order
    
    def _get_params_sell(self, new_step=False) -> Optional[Order]:
        order = None
        return order

    @property
    def buy_reasons(self) -> Set[str]:
        return {'Long', 'Short exit'}

    @property
    def sell_reasons(self) -> Set[str]:
        return {'Short', 'Long exit', 'Long stop', 'Long timeout'}
