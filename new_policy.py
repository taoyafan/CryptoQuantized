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
                    recovered = False
                # If not found the top and bottom check whether latest point is fake
                else:
                    if self.finding_bottom and self.highs[idx] > self.last_top and recovered == False:
                        # Found new top when searching bottom, last top is fake top, keep search top
                        self.finding_bottom = False
                        # Revert time info of last top
                        self.delta_time_top.recover()
                        self.last_top_time.recover()
                        self.last_checked_time -= 60000
                        recovered = True
                        
                    elif (not self.finding_bottom) and self.lows[idx] < self.last_bottom and (not recovered):
                        # Found new bottom when searching top, last bottom is fake bottom, keep search bottom
                        self.finding_bottom = True
                        # Revert time info of last bottom
                        self.delta_time_bottom.recover()
                        self.last_bottom_time.recover()
                        self.last_checked_time -= 60000
                        recovered = True
                    
                    else:
                        recovered = False

            # if idx >= self.front_threshold:
        # while confirmed_time <= timestamp:
        return

    def _get_params_buy(self, new_step=False) -> Order:
        idx = self.front_threshold + 1
        fake_top = np.max(self.highs[idx:]) if len(self.highs) > idx else 0
        return Order(OrderSide.BUY, max(self.last_top, fake_top), Order.ABOVE, 'Default',  # type: ignore
            self.account_state.get_timestamp())
    
    def _get_params_sell(self, new_step=False) -> Order:
        idx = self.front_threshold + 1
        fake_bottom = np.min(self.lows[idx:]) if len(self.lows) > idx else float('inf')
        return Order(OrderSide.SELL, min(self.last_bottom, fake_bottom), Order.BELLOW, 'Default', # type: ignore
            self.account_state.get_timestamp())

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
                self.last_bottom = self.lows[idx]
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
        # self.ma = MAs([10, 300])
        self.fee = kwargs['fee']
        self.last_ma300 = 0

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        self.atr.update(high - low)
        self.aer.update(close - self.last_close)
        # self.ma.update(close)

        last_top_time_temp    = self.last_top_time.value
        last_bottom_time_temp = self.last_bottom_time.value
        self.last_time        = timestamp
        self.last_close       = close

        super().update(high, low, close, volume, timestamp)

        # Whether cancel last order
        if (timestamp >= self.sell_order_valid_until and 
            self.sell_order and 
            self.sell_order.not_entered()
        ):
            self.account_state.cancel_order(self.sell_order)
            self.sell_order_valid_until = np.inf
        
        if (timestamp >= self.buy_order_valid_until and 
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
        if last_top_time_temp != self.last_top_time.value:
            self.can_buy = True
            self.new_top = True
        if last_bottom_time_temp != self.last_bottom_time.value:
            self.bottom_ordered = False

    def _get_params_buy(self, new_step=False) -> Optional[Order]:
        order = None

        if (new_step and
            self.atr.get_ma(60).valid() and
            self.can_buy == True
        ):

            atr60 = self.atr.get_ma(60).mean / self.last_close
            atr10 = self.atr.get_ma(10).mean / self.last_close
            aer10 = self.aer.get_ma(10).mean / self.last_close
            aer60 = self.aer.get_ma(60).mean / self.last_close
            
            buy_price = self.last_top
            # sell_atr = 1 - (atr10 - 0.000260) / (0.00228 - 0.000260) * 0.5
            sell_atr = 0.5
            sell_price = buy_price * (1 + sell_atr * atr10)
            stop_price = buy_price * (1 - sell_atr * atr10 / 2)
            p = 0.7
            possible_loss = buy_price - stop_price
            
            # rw = (sell_price - buy_price) / buy_price
            # rl = possible_loss / buy_price
            rw = (sell_price - buy_price) / buy_price - 2 * self.fee
            rl = possible_loss / buy_price + 2 * self.fee
            leverage = p / rl - (1 - p) / rw
            leverage = int(leverage // 5)
            # leverage = 1

            # dtop = (self.last_top - self.last_close) / self.last_close / atr60

            # ma300 = self.ma.get_ma(300).mean
            # ma_pos_pass = ma300 < self.last_bottom

            if rw > 0 and rl > 0 and leverage > 0: # and dtop < 2  and atr10 / atr60 < 1.5
            # if leverage > 0:
                open_time = self.account_state.get_timestamp()
                order = Order(OrderSide.BUY, self.last_top, Order.ABOVE, 'Long', 
                    open_time, leverage=leverage)
                # To make sure not move the order to finished
                # order.add_exit(0, Order.ABOVE, "Long timeout", lock_time=100*60000)
                order.add_exit(sell_price, Order.ABOVE, "Long exit")
                order.add_exit(stop_price, Order.BELLOW, "Long stop", lock_time=1*60000)

                open_time = self.account_state.get_timestamp()
                self.buy_order_valid_until = open_time      # current one step

        if self.new_top:
            # Each top only order once
            self.new_top = False

            if self.buy_order is not None and self.buy_order.not_entered():
                self.account_state.cancel_order(self.buy_order)

            # if self.atr.get_ma(300).valid() and self.atr.get_ma(10).mean > 0:
            # if self.atr.get_ma(10).valid() and self.atr.get_ma(10).mean > 0:
            #     atr10 = self.atr.get_ma(10).mean / self.last_close
                # atr300 = self.atr.get_ma(300).mean / self.last_close / atr10
                
                # max_100 = 1.3251994068033168 + 4.322486206153627 * atr300
                # dis_max = 1.0106702928240172 + 2.70972930087375 * atr300
                # min_100 = -2.498668601194724 + -2.7091676331450043 * atr300
                # dis_min = 1.1232896521059579 + 1.969042899166032 * atr300

                # buy_price = self.last_close
                # sell_price = buy_price * (1 + 3 * atr10)
                # stop_price = buy_price * (1 - 3 * atr10)
                # sell_price = buy_price * (1 + (max_100 - 1 * dis_max) * atr10)
                # stop_price = buy_price * (1 + (min_100 + 1 * dis_min) * atr10)

                # rw = (sell_price - buy_price) / buy_price
                # rl = (buy_price - stop_price) / buy_price
                # p = 0.6
                # times = p / rl - (1 - p) / rw
                # times = times // 1
                # times = 1
# 
                # if sell_price > buy_price and times > 0:
                    # open_time = self.account_state.get_timestamp()
                    # order = Order(OrderSide.BUY, max(self.last_close, self.last_top), Order.ABOVE, 'Long', 
                    #     open_time, leverage=times)

                    # if self.buy_order is not None and self.buy_order.wiat_exited():
                    #     self.buy_order.add_exit(sell_price, Order.ABOVE, "Long exit")
                    #     self.buy_order.add_exit(stop_price, Order.BELLOW, "Long stop")
                    # order.add_exit(0, Order.ABOVE, "Long timeout", lock_time=10*60000)
                    # self.buy_order_valid_until = open_time + 1*60000 # current and next

        return order
    
    def _get_params_sell(self, new_step=False) -> Optional[Order]:
        order = None

        # if (self.atr.get_ma(10).valid()  and 
        #     self.atr.get_ma(10).mean > 0 and
        #     self.buy_order is not None   and 
        #     self.buy_order.wiat_exited() and 
        #     self.buy_order.exits_num() == 1
        # ):
        #     atr10 = self.atr.get_ma(10).mean / self.last_close
        #     buy_price = self.last_close
        #     sell_price = buy_price * (1 + 3 * atr10)
        #     stop_price = buy_price * (1 - 3 * atr10)
            
        #     self.buy_order.add_exit(sell_price, Order.ABOVE, "Long exit")
        #     self.buy_order.add_exit(stop_price, Order.BELLOW, "Long stop")

        #     (not self.can_buy) and 
        #     (self.highs[-1] > self.last_top)
        # ):
        #     self.can_buy = True
        #     atr = self.atr.get_ma(self.atr_level).mean / self.last_close
        #     max = 1.0007571398183366 + 1.1536499457006657 * atr
        #     dis = 0.0006954201367866048 + 0.6535741678896724 * atr

        #     sell_price = self.last_close * (max + 3*dis)
        #     # sell_price = self.last_close
            
        #     order = Order(OrderSide.SELL, sell_price, Order.ABOVE, 'Short', 
        #         self.account_state.get_timestamp())
        #     order.add_exit(0, Order.ABOVE, "Short exit", lock_time=10*60000)
        #     self.sell_order_valid_until = self.last_time + 10*60000
        
        return order

    @property
    def buy_reasons(self) -> Set[str]:
        return {'Long', 'Short exit'}

    @property
    def sell_reasons(self) -> Set[str]:
        return {'Short', 'Long exit', 'Long stop', 'Long timeout'}


class PolicyBreakWithMa300Low(PolicyBreakThrough):

    def __init__(self, state: AccountState, time, log_en: bool=True, analyze_en: bool=True, policy_private_log: bool=False, **kwargs):
        super().__init__(state, time, log_en, analyze_en, policy_private_log, **kwargs)
        self.can_buy = True
        self.bottom_ordered = True
        self.last_time = time
        self.last_close = 0.0
        self.sell_order_valid_until = np.inf
        self.buy_order_valid_until = np.inf
        self.atr = MAs([10])
        self.ma = MAs([300])

    def update(self, high: float, low: float, close: float, volume: float, timestamp: int):
        last_top_time_temp    = self.last_top_time.value
        last_bottom_time_temp = self.last_bottom_time.value
        self.last_time        = timestamp
        self.last_close       = close

        super().update(high, low, close, volume, timestamp)

        
        self.atr.update(high-low)
        self.ma.update(close)

        # Whether cancel last order
        if (timestamp >= self.sell_order_valid_until and 
            self.sell_order and 
            self.sell_order.not_entered()
        ):
            self.account_state.cancel_order(self.sell_order)
            self.sell_order_valid_until = np.inf
        
        if (timestamp >= self.buy_order_valid_until and 
            self.buy_order and 
            self.buy_order.not_entered()
        ):
            self.account_state.cancel_order(self.buy_order)
            self.buy_order_valid_until = np.inf
        
        # Clear break up / down if top or bottom changed.
        if last_top_time_temp != self.last_top_time.value:
            self.can_buy = False
        if last_bottom_time_temp != self.last_bottom_time.value:
            self.bottom_ordered = False

    def _get_params_buy(self, new_step=False) -> Optional[Order]:
        order = None
        if (not self.can_buy            and
            self.highs[-1] >= self.last_top and
            new_step
        ):
            # Each top only order once
            self.can_buy = True

            if self.ma.get_ma(300).valid() and self.atr.get_ma(10).mean > 0:
                atr10 = self.atr.get_ma(10).mean / self.last_close
                ma300 = (self.ma.get_ma(300).mean - self.last_close) / self.last_close / atr10

                if ma300 < -4:
                    open_time = self.account_state.get_timestamp()
                    order = Order(OrderSide.BUY, 0, Order.ABOVE, 'Long', 
                        open_time, reduce_only=False)

        return order
    
    def _get_params_sell(self, new_step=False) -> Optional[Order]:
        order = None
        if (new_step and 
            (not self.bottom_ordered) and 
            (self.lows[-1] < self.last_bottom)
        ):
            self.bottom_ordered = True

            if (self.buy_order is not None) and (self.account_state.pos_amount > 0):
                order = Order(OrderSide.SELL, 0, Order.ABOVE, 'Short', 
                    self.account_state.get_timestamp(), reduce_only=True)
        
        return order

    @property
    def buy_reasons(self) -> Set[str]:
        return {'Long', 'Short exit'}

    @property
    def sell_reasons(self) -> Set[str]:
        return {'Short', 'Long exit', 'Long stop', 'Long timeout'}