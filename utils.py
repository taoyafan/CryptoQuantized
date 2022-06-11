import dateparser
import pytz
from datetime import datetime, timedelta
from typing import Union

from base_types import DataType

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
