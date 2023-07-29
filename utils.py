import dateparser
import pytz
from datetime import datetime, timedelta
from typing import Union

from base_types import DataType
import numpy as np
from typing import Dict, Optional, List, Set


class RingBuffer:
    def __init__(self, size: int) -> None:
        self.size = size
        self.buffer = np.zeros(size)
        self.num = 0
        self.idx = 0    # The next position to push
    
    def insert(self, data) -> int:
        out_data = 0
        if self.num < self.size:
            self.buffer[self.idx] = data
            self.idx += 1
            self.num += 1
        else:
            if self.idx == self.size:
                self.idx = 0
            out_data = self.buffer[self.idx]
            self.buffer[self.idx] = data
            self.idx += 1

        return out_data
    
    def get_latest_data(self):
        return self.buffer[self.idx - 1]
        
    def get_previous_data(self, previous_step: int) -> int:
        assert(previous_step >= 0 and previous_step < self.size)
        # previous_step: 0 Means current data.
        idx = self.idx - 1 - previous_step
        if idx < 0:
            idx = self.size + idx
        else:
            idx = idx

        return self.buffer[idx]


class MA:
    def __init__(self, level, buffer: RingBuffer):
        assert(level <= buffer.size)
        self.sum = 0
        self.level = level
        self.buffer = buffer

        self.mean = 0
        self.mean_pre_1 = 0
        self.mean_pre_2 = 0

    def update(self, new_data, insert_to_buffer=False):
        # 5 6 7  3 4     new data is 8, last data is 3 
        #     ^  ^
        #    idx |
        #     idx - 4
        last_data = self.buffer.get_previous_data(self.level-1)
        self.sum += new_data - last_data
        
        self.mean_pre_2 = self.mean_pre_1
        self.mean_pre_1 = self.mean
        self.mean = self.sum / min(self.level, self.buffer.num+1)

        if insert_to_buffer:
            self.buffer.insert(new_data)
    
    def get_latest_data(self):
        return self.buffer.get_latest_data()

    def valid(self) -> bool:
        return self.buffer.num >= self.level


class MAs:

    def __init__(self, ma_levels: List[int]) -> None:
        self.levels = ma_levels
        self.max_level = max(ma_levels)
        self.buffer = RingBuffer(self.max_level)
        self.mas : Dict[int, MA] = {}
        for level in ma_levels:
            self.mas[level] = (MA(level, self.buffer))
    
    def update(self, data):
        for level in self.mas.keys():
            self.mas[level].update(data)
        
        self.buffer.insert(data)
    
    def get_latest_data(self):
        return self.buffer.get_latest_data()
    
    def get_ma(self, level: int):
        return self.mas[level]


def date_to_milliseconds(date_str) -> int:
    """Convert UTC date to milliseconds

    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/

    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = dateparser.parse(date_str)

    # if the date is not timezone aware apply UTC timezone
    if d:
        if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
            d = d.replace(tzinfo=pytz.utc)

        # return the difference in time
        return int((d - epoch).total_seconds() * 1000.0)
    else:
        raise ValueError(date_str)

def milliseconds_to_date(ms: Union[int, float]) -> str:
    """Convert milliseconds to string of local data 
    e.g. 2022-05-21 22:24:18
    """
    epoch = datetime.fromtimestamp(0)
    return str(epoch + timedelta(milliseconds=ms))

def interval_to_milliseconds(interval: DataType) -> int:
    """Convert a Binance interval string to milliseconds

    :param interval: Binance interval string 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w
    :type interval: str

    :return:
        None if unit not one of m, h, d or w
        None if string not in correct format
        int value of interval in milliseconds
    """
    ms = None
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60
    }

    unit = interval.value[-1]
    if unit in seconds_per_unit:
        try:
            ms = int(interval.value[:-1]) * seconds_per_unit[unit] * 1000
        except ValueError:
            pass
    if ms:
        return ms
    else:
        raise ValueError(interval)
