from typing import Dict, Optional, List, Union
from enum import Enum

import time
import dateparser
import pytz
import os
import pandas as pd

from datetime import datetime, timedelta
from binance.client import Client

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

def milliseconds_to_date(ms: int) -> str:
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

def get_historical_klines(symbol, interval: DataType, start_ts, end_ts=None) -> pd.DataFrame:
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
        temp_data = client.get_klines(
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
        columns =['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                    'quote_assert_volume', 'number_of_trades', 'taker_buy_volume', 
                    'taker_buy_quote_asset_volume', 'ignore'])
    
    del output_data['ignore']
    # output_data = output_data.drop(columns=['ignore'])

    return output_data


class Data:

    def __init__(self, symbol: str, interval: DataType):
        self.symbol = symbol
        self.interval = interval
        self.file_loc = 'data/{}_{}.csv'.format(symbol, interval.value)
        self.data: pd.DataFrame = self._get_init_data()


    # ====================================== public ======================================

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
            klines = get_historical_klines(self.symbol, self.interval, start_ms, end_ms)
            
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
    symbol = "LUNABUSD"
    interval = DataType.INTERVAL_1MINUTE
    start = "2019/05/22 UTC+8"
    end = "1 minute ago UTC+8"

    data = Data(symbol, interval)
    data.update(start, end)
    print('Start time is {}'.format(data.start_time_str()))
    print('End time is {}'.format(data.end_time_str()))
    print('length {}'.format(data.len()))
    
if __name__ == "__main__":
    main()
    print("Finished")