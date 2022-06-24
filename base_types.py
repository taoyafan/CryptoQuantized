from typing import Dict, Optional, List, Set
from binance.client import Client
from enum import Enum, auto
import numpy as np 

class OrderSide(Enum):
    BUY = Client.SIDE_BUY
    SELL = Client.SIDE_SELL

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


class PolicyToAdaptor:
    class directionType(Enum):
        ABOVE = auto()
        BELLOW = auto()
    ABOVE = directionType.ABOVE
    BELLOW = directionType.BELLOW

    def __init__(self, price: float, direction: directionType, reason: str, 
                reduce_only: bool=False, is_order_market: bool=False):
        self.price: float = price
        self.direction: PolicyToAdaptor.directionType = direction
        self.reason = reason
        self.reduce_only = reduce_only
        self.is_order_market = is_order_market


# Buy or sell state
class OptState:

    # if reasons is for buy, then other_reasons is for sell
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
        self.pair_unfinished = False

    def add_part(self, idx: int, expect_price: float, actual_price: float, reason: str):
        assert self.pair_unfinished == False
        assert reason in self.reasons

        self._add_points(idx, expect_price, actual_price)
        self.last_reason = reason
        self.pair_unfinished = True
        
    def add_left_part(self, other_reason: str, earn: float):
        assert self.pair_unfinished and self.last_reason != None
        assert other_reason in self.other_reasons

        self.nums[self.last_reason][other_reason] += 1
        self.earns[self.last_reason][other_reason].append(earn)
        self.pair_unfinished = False

    def add_all(self, idx: int, expect_price: float, actual_price: float, reason: str, other_reason: str, earn: float):
        """ Add state of a finished option, i.e. buy or sell
        
        param:
            idx: Index of the option point
            value: Value of the option point
            reason: Option reason
            other_reason: The inverse option reason
            earn: earn amount / buy amount 
        """
        assert self.pair_unfinished == False, "Not allowed to add_part() and then add_all()"
        assert reason in self.reasons
        assert other_reason in self.other_reasons

        self._add_points(idx, expect_price, actual_price)
        self.nums[reason][other_reason] += 1
        self.earns[reason][other_reason].append(earn)
        self.pair_unfinished = False

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
                print('--- {} reason {}, nums: {}, earn nums: {}, average earn: {:.3f}%, median earn: {:.3f}%, max earn: {:.3f}%, min earn: {:.3f}%'.format(
                    other_name, o_r, self.nums[r][o_r], 
                    len([e for e in earns if e > 0]), np.mean(earns)*100, np.median(earns)*100, max(earns)*100, min(earns)*100
                ))
                earns_for_reason += earns
                nums_for_reason += self.nums[r][o_r]
            
            if len(self.other_reasons) > 1:
                print('- Total nums is {}, earn nums: {}, average earn: {:.3f}%, median earn: {:.3f}%, max earn: {:.3f}%, min earn: {:.3f}%'.format(
                    nums_for_reason, len([e for e in earns_for_reason if e > 0]), 
                    np.mean(earns_for_reason)*100, max(earns_for_reason)*100, np.median(earns_for_reason), min(earns_for_reason)*100
                ))
            earns_for_all += earns_for_reason
            nums_for_all += nums_for_reason

        if len(self.reasons) > 1:
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


