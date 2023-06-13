from __future__ import annotations
from typing import Dict, Optional, List, Set, TypeVar, Generic
from collections.abc import Callable
from binance.client import Client
from enum import Enum, auto
import numpy as np 

Value = TypeVar('Value')
class Recoverable(Generic[Value]):
    def __init__(self, value: Value):
        self._value: Value = value
        self._last_value: Value = value
    
    def set(self, value: Value):
        self._last_value = self.value
        self._value = value
    
    def recover(self):
        self._value = self._last_value

    @property
    def value(self):
        return self._value
    
    @property
    def last_value(self):
        return self._last_value

class OrderSide(Enum):
    BUY = Client.SIDE_BUY
    SELL = Client.SIDE_SELL

    def the_other(self):
        if self == OrderSide.BUY:
            return OrderSide.SELL
        else:
            return OrderSide.BUY

class IdxValue:
    def __init__(self, idx: Optional[List]=None, value: Optional[List]=None):
        self.idx = idx if idx else []
        self.value = value if value else []

    def add(self, idx: int, value):
        self.idx.append(idx)
        self.value.append(value)
    
    def is_empty(self) -> bool:
        return len(self.idx) == 0


class OptPoints:
    def __init__(self):
        self.buy = IdxValue()
        self.sell = IdxValue()


class DirectionType(Enum):
    ABOVE = auto()
    BELLOW = auto()


class TradeInfo:
    def __init__(self, price: float, direction: DirectionType, side: OrderSide, 
                 reason: str, reduce_only: bool, leverage: int = 1,
                 can_be_sent: bool = False, lock_time: int = 0):
        self.price: float             = price
        self.direction: DirectionType = direction
        self.side: OrderSide          = side
        self.reason: str              = reason
        self.reduce_only: bool        = reduce_only
        self.leverage: float          = leverage
        self.can_be_sent: bool        = can_be_sent
        
        # Info can be updated
        self.lock_start_time: Optional[int] = None
        self.lock_time: int = lock_time
        
        self._is_sent: bool = False
        self.client_order_id: Optional[int] = None

        self.executed_price: Optional[float] = None
        self.executed_time: Optional[int]    = None

    # TODO call this when executed
    def executed(self, executed_price: float, time: int):
        assert self.executed_price is None, "Can not set actual price twice"
        self.executed_price = executed_price
        self.executed_time = time

        if not self._is_sent:
            self._is_sent = True
    
    # TODO call this when sent
    def sent(self, client_order_id=None):
        assert self._is_sent == False, "Can not send twice"
        self._is_sent = True
        self.client_order_id = client_order_id

    def is_executed(self) -> bool:
        return self.executed_price is not None

    def is_sent(self) -> bool:
        return self._is_sent
    
    def set_locked_start_time(self, lock_start_time: int):
        self.lock_start_time = lock_start_time

    def locked_until(self) -> Optional[int]:
        # Return None if no lock
        return self.lock_start_time + self.lock_time   \
            if self.lock_start_time and self.lock_time \
            else None

    def is_locked(self, time: int) -> bool:
        is_locked = False
        if self.lock_time > 0: 
            if self.lock_start_time:
                is_locked = time < self.lock_start_time + self.lock_time
            else:
                # Haven't set a start time.
                is_locked = True

        return is_locked
    
    def equivalent_to(self, the_other: TradeInfo) -> bool:
        return (
            self.price          == the_other.price          and
            self.direction      == the_other.direction      and
            self.side           == the_other.side           and
            self.reason         == the_other.reason         and
            self.reduce_only    == the_other.reduce_only    and
            self.can_be_sent    == the_other.can_be_sent    and
            self._is_sent     == the_other._is_sent     and
            self.executed_price == the_other.executed_price and
            self.locked_until() == the_other.locked_until()
        )

class Priority():

    def __init__(self, value):
        assert(value >= 0)
        self.value = value
    
    def __lt__(self, others: 'Priority'):
        return self.value < others.value

    def __le__(self, others: 'Priority'):
        return self.value <= others.value

    def __gt__(self, others: 'Priority'):
        return self.value > others.value

    def __ge__(self, others: 'Priority'):
        return self.value >= others.value

    def __eq__(self, others: 'Priority'):
        return self.value == others.value

    def __ne__(self, others: 'Priority'):
        return self.value != others.value

class Order:
    ABOVE = DirectionType.ABOVE
    BELLOW = DirectionType.BELLOW

    class State(Enum):
        CREATED     = auto()    # Created this order request
        SENT      = auto()    # Sended this order to exchange
        ENTERED     = auto()    # Traded under the enter condition
        EXIT_SENT = auto()    # Sended exit condition to exchange
        EXITED      = auto()    # Traded under the exit condition
        FINISHED    = auto()    # Finished
        CANCELED    = auto()    # Canceled before finished

    def __init__(self, 
                 side: OrderSide,
                 price: float, 
                 direction: DirectionType, 
                 reason: str, 
                 create_time: int,
                 leverage: int = 1,
                 priority: Priority = Priority(2),
                 reduce_only: bool = False, 
                 can_be_sent: bool = False):
        """
        params:
            priority: Higher value higher priority

            cb_fun: Call back function after traded

            can_be_sent: whether send this order to exchange or hold until price meet the
                enter condition.
        """
        # Order params
        self.side: OrderSide = side
        self.state: Order.State = self.State.CREATED
        self.cancel_at_state: dict[Order.State, Order] = {}

        # Enter params
        self.entered_info = TradeInfo(price         = price,
                                      direction     = direction,
                                      side          = self.side,
                                      reason        = reason,
                                      reduce_only   = reduce_only,
                                      leverage      = leverage,
                                      can_be_sent = can_be_sent)

        self.create_time    = create_time
        self.enter_priority = priority
        self.reduce_only    = reduce_only
        
        # Exit params
        self.exited_infos: List[TradeInfo] = []
        self.exit_priority: Priority = Priority(0)       # None or the lowest exit priority

        # Call back functions
        self.cb_fun_traded: Optional[Callable[[TradeInfo], None]] = None
        self.cb_fun_canceled: Optional[Callable[[Order], bool]] = None

    def add_traded_call_back(self, cb_fun_traded: Callable[[TradeInfo], None]):
        self.cb_fun_traded = cb_fun_traded

    def add_canceled_call_back(self, cb_fun_canceled: Callable[[Order], bool]):
        self.cb_fun_canceled = cb_fun_canceled

    def not_entered(self) -> bool:
        return self.state.value < self.state.ENTERED.value
    
    def wiat_exited(self) -> bool:
        return self.is_alive() and self.state.value >= self.state.ENTERED.value

    def is_alive(self) -> bool:
        return self.state != Order.State.FINISHED and \
               self.state != Order.State.CANCELED

    def cancel(self) -> bool:
        canceled = False

        if (self.cb_fun_canceled and self.state != Order.State.FINISHED and
            self.state != Order.State.CANCELED
        ):
            canceled = self.cb_fun_canceled(self)

        return canceled

    def exits_num(self) -> int:
        return len(self.exited_infos)

    def has_exit(self) -> bool:
        return self.exits_num() > 0

    def num_exit_need_sent(self) -> int:
        return sum([int(info.can_be_sent) for info in self.exited_infos])

    def has_exit_need_sent(self) -> bool:
        return self.num_exit_need_sent() >= 1

    def clear_exit(self):
        self.exited_infos  = []
        self.exit_priority = Priority(0)

    # Exit can only be reduce only
    def add_exit(self, price: float, 
                 direction: DirectionType, 
                 reason: str,
                 priority: Priority = Priority(2),
                 can_be_sent: bool = False,
                 lock_time: int = 0):

        self.exited_infos.append(TradeInfo(price         = price,
                                           direction     = direction,
                                           side          = self.side.the_other(),
                                           reason        = reason,
                                           reduce_only   = True,
                                           can_be_sent   = can_be_sent,
                                           lock_time     = lock_time))
        # Lowest exit priority
        if self.exit_priority is None or priority < self.exit_priority:
            self.exit_priority = priority
    
    def set_state_to_next_traded(self):
        while self.state != Order.State.ENTERED or self.state != Order.State.EXITED:
            self._set_state(Order.State(self.state.value + 1))

    def set_state_to(self, state: State):
        assert state != Order.State.FINISHED, "Finished will be automatic set"
        assert state.value >= self.state.value, "State must move forward"

        if state == Order.State.CANCELED:
            self._set_state(state)
        while self.state != state:
            self._set_state(Order.State(self.state.value + 1))
        
        finished = self.state == Order.State.EXITED if self.has_exit() else \
                   self.state == Order.State.ENTERED
        if finished:
            self.state = Order.State.FINISHED
            assert len(self.cancel_at_state) == 0, \
                "Need to cancel all marked orders when finished"

    def _set_state(self, state: State):
        assert state != Order.State.FINISHED, "Finished will be automatic set"

        assert (state != Order.State.EXIT_SENT and \
                state != Order.State.EXITED) or \
               self.has_exit(), "Move to exited without exit condition" 

        assert state.value - self.state.value == 1 or \
               (state == Order.State.CANCELED and self.state != state), \
               "{} -> {}, Only allow one step increasing".format(self.state, state)

        if state in self.cancel_at_state:
            if (self.cancel_at_state[state].state != Order.State.CANCELED and \
                self.cancel_at_state[state].state != Order.State.FINISHED
            ):
                canceled = self.cancel_at_state[state].cancel()
                assert canceled, "Cancellation should be successful"
            del self.cancel_at_state[state]

        if state == self.State.ENTERED:
            assert self.entered_info.is_executed(), "Entered order must be executed"
            
            # Calling enter call back function
            if self.cb_fun_traded:
                self.cb_fun_traded(self.entered_info)

            # Set exit info
            entered_time = self.entered_info.executed_time
            if entered_time:
                for info in self.exited_infos:
                    info.set_locked_start_time(entered_time)
        
        elif state == self.State.EXITED:
            assert (self.has_exit() and \
                    sum([int(info.is_executed()) for info in self.exited_infos]) == 1), \
                   "There must be one exited order to be executed" 

            # Calling exit call back function
            if self.cb_fun_traded:
                for info in self.exited_infos:
                    if info.is_executed():
                        self.cb_fun_traded(info)
                        break

        self.state = state

    def cancel_another_at_state(self, state: State, another_order: 'Order'):
        self.cancel_at_state[state] = another_order

    def _exits_equivalent_to(self, order: Order) -> bool:
        is_exits_same = False
        num = self.exits_num()
        
        if num == order.exits_num():
            this = self.exited_infos
            other = order.exited_infos
            same_num = sum([int(this[i].equivalent_to(other[i])) for i in range(num)])

            if num == same_num:
                is_exits_same = True

        return is_exits_same

    def equivalent_to(self, order: 'Order') -> bool:
        is_enter_same = self.entered_info.equivalent_to(order.entered_info)
        is_exits_same = self._exits_equivalent_to(order)

        return (self.side            == order.side            and
                self.cancel_at_state == order.cancel_at_state and
                self.enter_priority  == order.enter_priority  and
                self.reduce_only     == order.reduce_only     and
                self.exit_priority   == order.exit_priority   and
                is_enter_same                                 and 
                is_exits_same)
                

    # def __del__(self):
    #     assert self.state == Order.State.FINISHED or \
    #            self.state == Order.State.CANCELED, "Must move state to finished before destory"

# Buy or sell state
class OptState:

    # if reasons is for buying, then other_reasons is for selling
    def __init__(self, reasons: Set[str], other_reasons: Set[str]):
        assert len(reasons) > 0 and len(other_reasons) > 0
        
        # Buy or sell points
        self.points_idx = []
        self.points_expect_price = []
        self.points_actual_price = []

        self.reasons = reasons
        self.other_reasons = other_reasons

        # Nums and earns for each option pair    
        self.nums: Dict[str, Dict[str, int]] = dict()           # nums[reason][other_reason]
        self.earns: Dict[str, Dict[str, List[float]]] = dict()    # nums[reason][other_reason][i]

        for r in reasons:
            self.nums[r] = dict()
            self.earns[r] = dict()
            for o_r in other_reasons:
                self.nums[r][o_r] = 0
                self.earns[r][o_r] = []
        
        # Temp value
        self.last_reason = ''
        self.has_added_part = False

    def reset(self):
        self.has_added_part = False

    def add_part(self, idx: int, expect_price: float, actual_price: float, reason: str, reduce_only=False):
        assert self.has_added_part == False
        assert reason in self.reasons

        self._add_points(idx, expect_price, actual_price)
        if not reduce_only:
            self.last_reason = reason
            self.has_added_part = True
        
    def add_left_part(self, other_reason: str, earn: float):
        assert self.has_added_part and self.last_reason != None
        assert other_reason in self.other_reasons
        # assert earn != 0

        self.nums[self.last_reason][other_reason] += 1
        self.earns[self.last_reason][other_reason].append(earn)
        self.has_added_part = False

    def add_all(self, idx: int, expect_price: float, actual_price: float, reason: str, other_reason: str, earn: float):
        """ Add state of a finished option, i.e. buy or sell
        
        param:
            idx: Index of the option point
            value: Value of the option point
            reason: Option reason
            other_reason: The inverse option reason
            earn: earn amount / buy amount 
        """
        assert self.has_added_part == False, "Not allowed to add_part() and then add_all()"
        assert reason in self.reasons
        assert other_reason in self.other_reasons

        self._add_points(idx, expect_price, actual_price)
        self.nums[reason][other_reason] += 1
        self.earns[reason][other_reason].append(earn)
        self.has_added_part = False

    def _add_points(self, idx, expect_price, actual_price):
        self.points_idx.append(idx)
        self.points_expect_price.append(expect_price)
        self.points_actual_price.append(actual_price)

    def log(self, name: str, other_name: str):
        
        earns_for_all = []
        nums_for_all = 0
        for r in self.reasons:
            earns_for_reason = []
            nums_for_reason = 0
            print('- For {} reason {}: '.format(name, r))

            for o_r in self.other_reasons:
                earns = self.earns[r][o_r]
                if len(earns) > 0:
                    print('--- {} reason {}, nums: {}, earn nums: {}, average earn: {:.3f}%, median earn: {:.3f}%, max earn: {:.3f}%, min earn: {:.3f}%'.format(
                        other_name, o_r, self.nums[r][o_r], 
                        len([e for e in earns if e >= 0]), np.mean(earns)*100, np.median(earns)*100, max(earns)*100, min(earns)*100
                    ))
                earns_for_reason += earns
                nums_for_reason += self.nums[r][o_r]
            
            if len(self.other_reasons) > 1 and len(earns_for_reason) > 0:
                earn_num = len([e for e in earns_for_reason if e >= 0])
                print('- Total nums is {}, earn nums: {}, {:.2f}%, average earn: {:.3f}%, median earn: {:.3f}%, max earn: {:.3f}%, min earn: {:.3f}%'.format(
                    nums_for_reason, earn_num, 100*earn_num/nums_for_reason, 
                    np.mean(earns_for_reason)*100, np.median(earns_for_reason)*100, max(earns_for_reason)*100, min(earns_for_reason)*100
                ))
            earns_for_all += earns_for_reason
            nums_for_all += nums_for_reason

        if len(self.reasons) > 1 and len(earns_for_all) > 0:
            print('Total nums is {}, earn nums: {}, average earn: {:.3f}%, median earn: {:.3f}%, max earn: {:.3f}%, min earn: {:.3f}%'.format(
                nums_for_all, len([e > 0 for e in earns_for_all]), 
                    np.mean(earns_for_all)*100, max(earns_for_all)*100, np.median(earns_for_all), min(earns_for_all)*100
            ))


class DataElements(Enum):
    # The order MUST be same as the API returns
    OPEN_TIME = 'open_time'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'
    CLOSE = 'close'
    VOLUME = 'volume'
    CLOSE_TIME = 'close_time'
    QUOTE_ASSERT_VOLUME = 'quote_assert_volume'
    NUMBER_OF_TRADES = 'number_of_trades'
    TAKER_BUY_VOLUME = 'taker_buy_volume'
    TAKER_BUY_QUOTE_ASSET_VOLUME = 'taker_buy_quote_asset_volume'

class DataType(Enum):
    INTERVAL_1MINUTE = Client.KLINE_INTERVAL_1MINUTE
    INTERVAL_3MINUTE = Client.KLINE_INTERVAL_3MINUTE
    INTERVAL_5MINUTE = Client.KLINE_INTERVAL_5MINUTE
    INTERVAL_15MINUTE = Client.KLINE_INTERVAL_15MINUTE
    INTERVAL_30MINUTE = Client.KLINE_INTERVAL_30MINUTE
    INTERVAL_1HOUR = Client.KLINE_INTERVAL_1HOUR
    INTERVAL_2HOUR = Client.KLINE_INTERVAL_2HOUR
    INTERVAL_4HOUR = Client.KLINE_INTERVAL_4HOUR
    INTERVAL_6HOUR = Client.KLINE_INTERVAL_6HOUR
    INTERVAL_8HOUR = Client.KLINE_INTERVAL_8HOUR
    INTERVAL_12HOUR = Client.KLINE_INTERVAL_12HOUR
    INTERVAL_1DAY = Client.KLINE_INTERVAL_1DAY
    INTERVAL_3DAY = Client.KLINE_INTERVAL_3DAY
    INTERVAL_1WEEK = Client.KLINE_INTERVAL_1WEEK
    INTERVAL_1MONTH = Client.KLINE_INTERVAL_1MONTH


