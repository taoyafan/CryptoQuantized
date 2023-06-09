from typing import Dict, Optional, List, Union
from enum import Enum
import time
import os
import pandas as pd

from binance.client import Client
from utils import *
from base_types import DataType, DataElements


def get_historical_klines(client: Client, symbol, interval: DataType, start_ts, end_ts=None, 
                          is_futures: Optional[bool]=False) -> pd.DataFrame:
    """Get Historical Klines from Binance

    See dateparse docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/

    If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

    :param symbol: Name of symbol pair e.g BNBBTC
    :type symbol: str
    :param interval: Biannce Kline interval
    :type interval: str
    :param start_ts:
    :type start_ts: int
    :param end_ts:
    :type end_ts: int

    :return: pd.DataFrame

    """

    # init our list
    output_data = []

    # setup the max limit
    limit = 500

    # convert interval to useful value in seconds
    timeframe = interval_to_milliseconds(interval)

    idx = 0
    total = (end_ts - start_ts) // (timeframe * limit) + 1
    # it can be difficult to know when a symbol was listed on Binance so allow start time to be before list date
    symbol_existed = False
    while True:
        # fetch the klines from start_ts up to max 500 entries or the end_ts if set
        fun = client.futures_klines if is_futures else client.get_klines
        temp_data = fun(
            symbol=symbol,
            interval=interval.value,
            limit=limit,
            startTime=start_ts,
            endTime=end_ts
        )

        # handle the case where our start date is before the symbol pair listed on Binance
        if not symbol_existed and len(temp_data):
            symbol_existed = True

        if symbol_existed:
            # append this loops data to our output data
            output_data += temp_data

            # update our start timestamp using the last value in the array and add the interval timeframe
            start_ts = temp_data[len(temp_data) - 1][0] + timeframe
        else:
            # it wasn't listed yet, increment our start date
            start_ts += timeframe

        idx += 1
        if total > 5:
            print('Idx: {} / {}'.format(idx, total), end='\r')
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)
        
    if total > 5:
        print()
    output_data = pd.DataFrame(
        output_data, columns = [e.value for e in DataElements] + ['ignore'])
    
    del output_data['ignore']
    # output_data = output_data.drop(columns=['ignore'])

    return output_data


class Data:

    def __init__(self, symbol: str, interval: DataType, start_str: Optional[str]=None, 
                 end_str: Optional[str]=None, start_idx: Optional[int]=None, 
                 end_idx: Optional[int]=None, num: Optional[int]=None, 
                 is_futures: Optional[bool]=False):
        """
        params:
            MUST set:
                symbol: e.g. BTCBUSD
                interval: e.g. DataType.INTERVAL_1MINUTE
            
            When set range of data, one of two type pairs can choose:
                start_str, end_str(Optional): e.g. "2021/3/1 UTC+8", "2021/5/1 UTC+8"
                start_idx, end_idx(Optional): e.g. 100, 200

                If num is not set, start must set. end is optional and the default origin end.
                If num is set. select num of data from start or end of end(default). Can not set both
                    start and end.
            Others:
                is_futures: is data is futures
        """
        self.client = None
        self.symbol = symbol
        self.interval = interval
        self.interval_ms = interval_to_milliseconds(self.interval)
        self.is_futures = is_futures
        
        # Get the initial data from file if has
        self.file_loc = 'data/{}_{}{}.csv'.format(symbol, interval.value, 
            '_futures' if is_futures else '')
        self.data: pd.DataFrame = self._get_init_data()
        
        # Whether can call update method
        self.can_update = True
        if start_str or end_str or start_idx or end_idx or num:
            self.replace_data_with_range(start_str, end_str, start_idx, end_idx, num)

        self.time2index = pd.DataFrame()


    # ====================================== public ======================================

    def set_client(self, client: Client):
        self.client = client

    # May change self.data
    def replace_data_with_range(self, start_str: Optional[str]=None, end_str: Optional[str]=None, 
                                start_idx: Optional[int]=None, end_idx: Optional[int]=None, 
                                num: Optional[int]=None):
        """Get the range of data with which replace self.data

            One of two type pairs can choose:
                start_str, end_str(Optional): e.g. "2021/3/1 UTC+8", "2021/5/1 UTC+8"
                start_idx, end_idx(Optional): e.g. 100, 200

                If num is not set, start must set. end is optional and the default origin end.
                If num is set. select num of data from start or end of end(default). Can not set both
                    start and end.
        """
        # 1. Convert str to idx
        if start_str or end_str:
            assert not (start_idx or end_idx), 'Can not set both date str and idx'
            start_ms = date_to_milliseconds(start_str) if start_str else None
            end_ms = date_to_milliseconds(end_str) if end_str else None
            
            data = self.data
            if start_ms:
                data = data[(data[DataElements.OPEN_TIME.value] >= start_ms)]
                start_idx = data.index.values[0] if len(data) > 0 else None 

            if end_ms:
                data = data[(data[DataElements.CLOSE_TIME.value] <= end_ms)]
                end_idx = data.index.values[-1] + 1 if len(data.index) > 0 else None 
    
        # print('After convert str to idx, start idx: {}, end idx: {}'.format(start_idx, end_idx))
        # 2. Set both start_idx and end_idx
        if num:
            assert not (start_idx and end_idx), 'Can not set both start and end when set num'
            if start_idx is not None:
                if start_idx < 0:
                    start_idx = len(self.data) - start_idx
                end_idx = start_idx + num
            else:
                # start_idx is None
                if end_idx is None or end_idx >= len(self.data):
                    end_idx = len(self.data)
                elif end_idx < 0:
                    end_idx = len(self.data) - end_idx
                
                start_idx = end_idx-num

        # 3. Check out bound and get the target data
        # start_idx will not be None
        if start_idx is None or start_idx < 0:
            start_idx = 0

        if end_idx and end_idx >= len(self.data):
            end_idx = None
        
        if end_idx:
            self.can_update = False

        # print('Start idx: {}, end idx: {}'.format(start_idx, end_idx))
        self.data = self.data.iloc[start_idx:end_idx, :].reset_index(drop=True)
        
        
    def update(self, start_str: Optional[str]=None, end_str: Optional[str]=None, 
               start_ms: Optional[int]=None, end_ms: Optional[int]=None) -> bool:
        """Update slef.data from start_str to end_str
            
        params: 
            start_str:  start date in readable format, 
                        i.e. "January 01, 2018", "11 hours ago UTC", "now UTC".
                        None means start time is self.end_time + 1
            end_str:    end date in readable format. None means now.
        
        return: Whether updated
        """
        is_updated = False
        if self.can_update:
            if start_str or end_str:
                assert start_ms is None and end_ms is None, 'Input type can only be str or int(ms)'
                start_ms = date_to_milliseconds(start_str) if start_str else None
                end_ms = date_to_milliseconds(end_str) if end_str else None
            
            start_ms, end_ms = self._get_best_time(start_ms, end_ms)
            
            if end_ms > start_ms:
                if self.client is None:
                    proxies = {
                        "http": "http://127.0.0.1:8900",
                        "https": "http://127.0.0.1:8900",
                    }

                    # create the Binance client, no need for api key
                    self.client = Client("", "",  {'proxies': proxies})

                klines = get_historical_klines(self.client, self.symbol, self.interval, 
                                            start_ms, end_ms, self.is_futures)
                
                if len(klines) > 0:
                    if os.path.exists(self.file_loc):
                        # If file exist, append.
                        klines.to_csv(self.file_loc, mode='a', index=False, header=False)
                    else:
                        klines.to_csv(self.file_loc, index=False)
                    
                    self.data = self.data.append(klines)
                    is_updated = True
            else:
                # end <= start, no need to update
                pass
        else:
            # End of data changed. Can not update.
            pass

        return is_updated

    def start_time_str(self) -> str:
        start_ms = self.start_time()
        if start_ms:
            start_str = milliseconds_to_date(start_ms)
        else:
            start_str = 'No data'
        return start_str

    def start_time(self) -> Union[int, None]:
        if len(self.data) > 0:
            start = int(self.data['open_time'].values[0])  # type: ignore
        else:
            start = None
        
        return start

    def end_time_str(self) -> str:
        end_ms = self.end_time()
        if end_ms:
            end_str = milliseconds_to_date(end_ms)
        else:
            end_str = 'No data'
        return end_str

    def end_time(self) -> Union[int, None]:
        if len(self.data) > 0:
            end = int(self.data['close_time'].values[-1])  # type: ignore
        else:
            end = None
        
        return end

    def len(self) -> int:
        return len(self.data)
    
    def get_columns(self, names: List[DataElements]):
        return [self.data[n.value] for n in names]

    def get_value(self, name: DataElements, i: int) -> float:
        assert i < self.len()
        return float(self.data[name.value].values[i])  # type: ignore

    def time_list_to_idx(self, time_list:List) -> List:
        self._update_time2idx()
        for i in range(len(time_list)):
            time_list[i] = self.time2index.loc[time_list[i] // self.interval_ms * self.interval_ms, 'index']
        return time_list

    def time_to_idx(self, time: int):
        return self.time2index.loc[time // self.interval_ms * self.interval_ms, 'index']

    # ====================================== internal ======================================

    def _update_time2idx(self):
        if len(self.time2index) != len(self.data):
            self.time2index = self.data[DataElements.OPEN_TIME.value]
            self.time2index = self.time2index.reset_index()
            self.time2index = self.time2index.set_index(DataElements.OPEN_TIME.value)

    def _get_init_data(self) -> pd.DataFrame:
        if os.path.exists(self.file_loc):
            data = pd.read_csv(self.file_loc)
        else:
            data = pd.DataFrame()

        return data
        
    def _get_best_time(self, start_ms: Optional[int], end_ms: Optional[int]):
        # If not set end_ms, default is now
        if end_ms is None:
            end_ms = date_to_milliseconds('now')

        last_end = self.end_time()
        if last_end:
            # Date is exist before, start is next time of saved data.
            start_ms = last_end + 1
        else:
            if not start_ms:
                # Date is not exist, get 1000 data
                total_ms = 1000 * interval_to_milliseconds(self.interval)
                start_ms = end_ms + 1 - total_ms
                print('Not set the start, default get 1000 data before now')
        
        return start_ms, end_ms


def main():
    # symbol = "BTCUSDT"
    # symbol = "BTCBUSD"
    symbol = "BTCTUSD"
    # symbol = "SOLBUSD"
    # symbol = "GMTBUSD"
    # symbol = "DOGEBUSD"
    # symbol = "1000LUNCBUSD"
    interval = DataType.INTERVAL_1MINUTE
    start = "2023/3/23 00:00 UTC+8"
    # start = "140 days ago UTC+8"
    end = "1 minute ago UTC+8"

    data = Data(symbol, interval, is_futures=False)
    # data = Data(symbol, interval)
    data.update(start, end)
    # data.replace_data_with_date("2019/8/23 UTC+8", "2020/3/1 UTC+8")
    print('Start time is {}'.format(data.start_time_str()))
    print('End time is {}'.format(data.end_time_str()))
    print('length {}'.format(data.len()))
    
if __name__ == "__main__":
    main()
    print("Finished")