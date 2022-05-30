from typing import Dict, Optional, List, Set
from binance.client import Client
from enum import Enum

class IdxValue:
    def __init__(self, idx: Optional[List]=None, value: Optional[List]=None):
        self.idx = idx if idx else []
        self.value = value if value else []
    
    def add(self, idx, value):
        self.idx.append(idx)
        self.value.append(value)

class OptPoints:
    def __init__(self):
        self.buy = IdxValue()
        self.sell = IdxValue()


# Buy or sell state
class OptState:

    # if reasons is for buy, then other_reasons is for sell
    def __init__(self, reasons: Set[str], other_reasons: Set[str]):
        assert len(reasons) > 0 and len(other_reasons) > 0
        
        self.points = IdxValue()    # Buy or sell points

        self.reasons = reasons
        self.other_reasons = other_reasons

        # Nums and earns for each option pair    
        self.nums: Dict[str, Dict[str, int]] = dict()           # nums[reason][other_reason]
        self.earns: Dict[str, Dict[str, List[int]]] = dict()    # nums[reason][other_reason][i]

        for r in reasons:
            for o_r in other_reasons:
                self.nums[r][o_r] = 0
                self.earns[r][o_r] = []
        
        # Temp value
        self.last_reason = None
        self.pair_unfinished = False

    def add_part(self, idx, value, reason):
        assert self.pair_unfinished == False

        self.points.add(idx, value)
        self.last_reason = reason
        self.pair_unfinished = True
        
    def add_left_part(self, other_reason, earn):
        assert self.pair_unfinished and self.last_reason != None

        self.nums[self.last_reason][other_reason] += 1
        self.earns[self.last_reason][other_reason].append(earn)
        self.pair_unfinished = False


    def add_all(self, idx, value, reason, other_reason, earn):
        """ Add state of a finished option, i.e. buy or sell
        
        param:
            idx: Index of the option point
            value: Value of the option point
            reason: Option reason
            other_reason: The inverse option reason
            earn: earn amount / buy amount 
        """
        assert self.pair_unfinished == False, "Not allowed to add_part() and then add_all()"

        self.points.add(idx, value)
        self.nums[reason][other_reason] += 1
        self.earns[reason][other_reason].append(earn)
        self.pair_unfinished = False

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