from typing import Dict, Optional, List
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance
import numpy as np
from matplotlib import ticker

from base_types import IdxValue, OptPoints

class PricePlot:

    class Points:
        def __init__(self, idx, value, s, c, label):
            self.idx = idx
            self.value = value
            self.s = s
            self.c = c
            self.label = label

    def __init__(self, open, high, low, close, open_time):
        self.data = pd.DataFrame()
        self.data['open'] = open
        self.data['high'] = high
        self.data['low'] = low
        self.data['close'] = close
        self.data['timestamp'] = open_time
        self.data = self.data.reset_index(drop=True)

    def plot(self, plot_candle=True, plot_vol=False, datum=None, datum_lines=None, 
             opt_point: Optional[OptPoints]=None, earn_point: Optional[IdxValue]=None, 
             points: Optional[List[Points]]=None):

        figure_num = 1
        figure_num += 1 if plot_vol else 0
        figure_num += 1 if earn_point else 0
        if figure_num == 0:
            return

        num = 0
        subplot = plt.subplot2grid((figure_num, 1), (num, 0), rowspan=1, colspan=1)
        if plot_candle:
            self._candle_plot(subplot)
        else:
            subplot.plot(range(0, len(self.data)), self.data['close'],  # type: ignore
                        color="gray", linewidth=1.0, label='base')
                        
        for ma in [60, 240]:
            subplot.plot(range(0, len(self.data)), self.data['close'].rolling(ma).mean())

        if points:
            for p in points:
                subplot.scatter(p.idx, p.value, s=p.s, c=p.c, label=p.label)  # type: ignore
        if opt_point:
            subplot.scatter(opt_point.buy.idx, opt_point.buy.value, s=90, c='r', label="buy")  # type: ignore
            subplot.scatter(opt_point.sell.idx, opt_point.sell.value, s=90, c='g', label="sell") # type: ignore
        if datum:
            subplot.scatter(datum['high']['idx'], datum['high']['value'], s=30, c='b', label="high")  # type: ignore
            subplot.scatter(datum['low']['idx'], datum['low']['value'], s=30, c='y', label="low") # type: ignore
        if datum_lines:
            self._datum_lines_plot(subplot, datum_lines)
        # plt.legend()
        num += 1

        if plot_vol:
            subplot = plt.subplot2grid((figure_num, 1), (num, 0), rowspan=1, colspan=1, sharex=subplot)
            self._vol_plot(subplot)
            num += 1
        
        if earn_point:
            subplot = plt.subplot2grid((figure_num, 1), (num, 0), rowspan=1, colspan=1, sharex=subplot)
            self._earn_plot(subplot, earn_point)
        
        if subplot:
            def format_date(x, pos):
                if x < 0 or x > self.data.shape[0] - 1:
                    return ''
                return self.data.timestamp.values[int(x)]
            subplot.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
            
        # plt.legend()
        plt.show()

    def _candle_plot(self, subplot):
        self.data['dates'] = np.arange(0, len(self.data))
        ohlc = self.data[['open', 'high', 'low', 'close']]
        ohlc = ohlc.reset_index()

        mpl_finance.candlestick_ohlc(ax=subplot, quotes=ohlc.values,
                                     width=0.7, colorup='g', colordown='r')

        subplot.set_ylabel('Price')
        return

    def _datum_lines_plot(self, subplot, datum_lines):
        color = {'high_trend_line': 'red',
                 'parallel_low_trend_line': 'pink',
                 'high_horizontal_line': 'yellow',
                 'low_trend_line': 'green',
                 'parallel_high_trend_line': 'blue',
                 'low_horizontal_line': 'lightgreen'}
        for key in datum_lines:
            label_en = True
            for i_v in datum_lines[key]:
                subplot.plot(i_v[0], i_v[1], linewidth=1.0, color=color[key], label=key if label_en else None)
                label_en = False

    def _vol_plot(self, subplot):
        self.data['up'] = self.data.apply(lambda row: 1 if row['close'] >= row['open'] else 0, axis=1)
        subplot.bar(self.data.query('up == 1')['dates'].values,
                    self.data.query('up == 1')['volume'].values / 10000, color='g')  # type: ignore
        subplot.bar(self.data.query('up == 0')['dates'].values,
                    self.data.query('up == 0')['volume'].values / 10000, color='r')  # type: ignore
        subplot.set_ylabel('vol (W)')
        return

    def _earn_plot(self, subplot, earn_point: IdxValue):
        subplot.plot(range(0, self.data.shape[0]), self.data.close / self.data.close[0],
                     color="blue", linewidth=1.0, label='base')
        subplot.plot(earn_point.idx + [self.data.shape[0] - 1], earn_point.value + [earn_point.value[-1]],
                     color="red", linewidth=1.0, label='policy')

def main():
    luna = Data('LUNABUSD', DataType.INTERVAL_1MINUTE).data.iloc[-300:, :].reset_index(drop=True)
    fig = PricePlot(luna['open'], luna['high'], luna['low'], luna['close'], 
        luna['open_time'].map(milliseconds_to_date))
    fig.plot()

if __name__ == "__main__":
    from data import Data, DataType, milliseconds_to_date
    main()
    print("Finished")