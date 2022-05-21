from typing import Dict, Optional, List
from enum import Enum, unique

import time
import dateparser
import pytz
import json
import pandas as pd

from datetime import datetime, timedelta
from binance.client import Client

@unique
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


class Data:

    def __init__(self):
        self.data: Dict[DataType, pd.DataFrame] = self._get_init_data()

    def get_data(self, symbol: str, interval: DataType, start: str, end: str=None):
        klines = self._get_historical_klines(symbol, interval, start, end)

        # open a file with filename including symbol, interval and start and end converted to milliseconds
        with open(
            "Binance_{}_{}_{}-{}.json".format(
                symbol,
                interval,
                self._date_to_milliseconds(start),
                self._date_to_milliseconds(end)
            ),
            'w'  # set file write mode
        ) as f:
            f.write(json.dumps(klines))

    # TODO read data from file if exist
    def _get_init_data():
        return pd.DataFrame()

    def _date_to_milliseconds(self, date_str):
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
        if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
            d = d.replace(tzinfo=pytz.utc)

        # return the difference in time
        return int((d - epoch).total_seconds() * 1000.0)
    
    def _milliseconds_to_date(self, ms: int) -> str:
        # get epoch value in UTC
        epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
        return str(epoch + timedelta(milliseconds=ms))

    def _interval_to_milliseconds(self, interval: DataType):
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
        return ms

    def _get_historical_klines(self, symbol, interval: DataType, start_str, end_str=None):
        """Get Historical Klines from Binance

        See dateparse docs for valid start and end string formats http://dateparser.readthedocs.io/en/latest/

        If using offset strings for dates add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"

        :param symbol: Name of symbol pair e.g BNBBTC
        :type symbol: str
        :param interval: Biannce Kline interval
        :type interval: str
        :param start_str: Start date string in UTC format
        :type start_str: str
        :param end_str: optional - end date string in UTC format
        :type end_str: str

        :return: list of OHLCV values

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
        timeframe = self._interval_to_milliseconds(interval)

        # convert our date strings to milliseconds
        start_ts = self._date_to_milliseconds(start_str)

        # if an end time was passed convert it
        end_ts = None
        if end_str:
            end_ts = self._date_to_milliseconds(end_str)

        idx = 0
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
            # check if we received less than the required limit and exit the loop
            if len(temp_data) < limit:
                # exit the while loop
                break

            # sleep after every 3rd call to be kind to the API
            if idx % 3 == 0:
                time.sleep(1)

        return output_data


def main():
    symbol = "BTCUSDT"
    interval = DataType.INTERVAL_1DAY
    start = "10 days ago"
    end = "now"
    get_and_save_data(symbol, interval, start, end)

    
if __name__ == "__main__":
    main()
    print("Finished")