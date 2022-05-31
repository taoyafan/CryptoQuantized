from typing import Dict, Optional, List, Union

import time
import os
import pandas as pd

from binance.client import Client
from utils import *
from base_types import DataType

class DataColumns():
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


def get_historical_klines(symbol, interval: DataType, start_ts, end_ts=None, 
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
    proxies = {
        "http": "http://127.0.0.1:8900",
        "https": "http://127.0.0.1:8900",
    }

    # create the Binance client, no need for api key
    client = Client("", "",  {'proxies': proxies})

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
        print('Idx: {} / {}'.format(idx, total), end='\r')
        # check if we received less than the required limit and exit the loop
        if len(temp_data) < limit:
            # exit the while loop
            break

        # sleep after every 3rd call to be kind to the API
        if idx % 3 == 0:
            time.sleep(1)
        
    print()
    output_data = pd.DataFrame(
        output_data, 
        columns =[
            DataColumns.OPEN_TIME, DataColumns.OPEN, DataColumns.HIGH, DataColumns.LOW, 
            DataColumns.CLOSE, DataColumns.VOLUME, DataColumns.CLOSE_TIME, 
            DataColumns.QUOTE_ASSERT_VOLUME, DataColumns.NUMBER_OF_TRADES, 
            DataColumns.TAKER_BUY_VOLUME, DataColumns.TAKER_BUY_QUOTE_ASSET_VOLUME, 'ignore'])
    
    del output_data['ignore']
    # output_data = output_data.drop(columns=['ignore'])

    return output_data


class Data:

    def __init__(self, symbol: str, interval: DataType, start_str: Optional[str]=None, 
                 end_str: Optional[str]=None, is_futures: Optional[bool]=False):
        self.symbol = symbol
        self.interval = interval
        self.is_futures = is_futures
        self.file_loc = 'data/{}_{}{}.csv'.format(symbol, interval.value, 
            '_futures' if is_futures else '')
        self.data: pd.DataFrame = self._get_init_data()
        if start_str:
            self.replace_data_with_date(start_str, end_str)


    # ====================================== public ======================================

    def replace_data_with_date(self, start_str: str, end_str: Optional[str]=None):
        interval_ms = interval_to_milliseconds(self.interval)
        
        # Get the start idx
        start_of_all = self.start_time()
        start_ms = date_to_milliseconds(start_str)
        if start_of_all:
            if start_ms >= start_of_all:
                start_idx = (start_ms - start_of_all) // interval_ms
            else:
                start_idx = 0
        else:
            start_idx = None
        
        # Get the end idx
        end_of_all = self.end_time()
        if end_str:
            end_ms = date_to_milliseconds(end_str) 
            if end_ms and end_of_all and end_ms > end_of_all:
                end_ms = end_of_all
        else:
            end_ms = end_of_all
        
        if end_ms and start_of_all:
            end_idx = (end_ms - start_of_all) // interval_ms + 1
        else:
            end_idx = None
        
        # Get data
        if start_idx and end_idx:
            data = self.data.iloc[start_idx: end_idx, :].reset_index(drop=True)
        else:
            data = pd.DataFrame()
        
        self.data = data
        
    def update(self, start_str: Optional[str]=None, end_str: Optional[str]=None):
        """Update slef.data from start_str to end_str
            
        params: 
            start_str:  start date in readable format, 
                        i.e. "January 01, 2018", "11 hours ago UTC", "now UTC".
                        None means start time is self.end_time + 1
            end_str:    end date in readable format. None means now.
        """
        
        start_ms, end_ms = self._time_str_to_ms(start_str, end_str)
        
        if end_ms > start_ms:
            klines = get_historical_klines(self.symbol, self.interval, 
                                           start_ms, end_ms, self.is_futures)
            
            if len(klines) > 0:
                if os.path.exists(self.file_loc):
                    # If file exist, append.
                    klines.to_csv(self.file_loc, mode='a', index=False, header=False)
                else:
                    klines.to_csv(self.file_loc, index=False)
                
                self.data = self.data.append(klines)
        else:
            print('end <= start, no need to update')

    def start_time_str(self) -> str:
        start_ms = self.start_time()
        if start_ms:
            start_str = milliseconds_to_date(start_ms)
        else:
            start_str = 'No data'
        return start_str

    def start_time(self) -> Union[int, None]:
        if len(self.data) > 0:
            start = int(self.data['open_time'].values[0])
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
            end = int(self.data['close_time'].values[-1])
        else:
            end = None
        
        return end

    def len(self) -> int:
        return len(self.data)

    # ====================================== internal ======================================

    def _get_init_data(self) -> pd.DataFrame:
        if os.path.exists(self.file_loc):
            data = pd.read_csv(self.file_loc)
        else:
            data = pd.DataFrame()

        return data
        
    def _time_str_to_ms(self, start_str, end_str):
        # if an end time was passed convert it
        if end_str:
            end_ms = date_to_milliseconds(end_str)
        else:
            end_ms = date_to_milliseconds('now')

        # convert our date strings to milliseconds
        last_end = self.end_time()
        if last_end:
            # Date is exist before, start is next time of saved data.
            start_ms = last_end + 1
        else:
            if start_str:
                start_ms = date_to_milliseconds(start_str)
            else:
                # Date is not exist, get 1000 data
                total_ms = 1000 * interval_to_milliseconds(self.interval)
                start_ms = end_ms + 1 - total_ms
        
        return start_ms, end_ms


def main():
    # symbol = "BTCUSDT"
    symbol = "BTCBUSD"
    # symbol = "LUNABUSD"
    # symbol = "LUNCBUSD"
    interval = DataType.INTERVAL_1MINUTE
    # start = "2022/05/31 14:11 UTC+8"
    start = "140 days ago UTC+8"
    end = "1 minute ago UTC+8"

    # data = Data(symbol, interval, is_futures=True)
    data = Data(symbol, interval)
    data.update(start, end)
    # data.replace_data_with_date("2019/8/23 UTC+8", "2020/3/1 UTC+8")
    print('Start time is {}'.format(data.start_time_str()))
    print('End time is {}'.format(data.end_time_str()))
    print('length {}'.format(data.len()))
    
if __name__ == "__main__":
    main()
    print("Finished")