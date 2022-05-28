from typing import List
from enum import Enum, auto
from plot import PricePlot

class TrendLine:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.k = (self.y1 - self.y0) / (self.x1 - self.x0) if self.x1 != self.x0 else float('inf')

    def update(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.k = (self.y1 - self.y0) / (self.x1 - self.x0) if self.x1 != self.x0 else float('inf')

    def predict(self, x):
        if self.k == float('inf'):
            raise ValueError
        else:
            return self.y0 + (x - self.x0) * self.k

    def values(self):
        # 只对画图有用
        if self.k == 0:
            x1 = self.x0 + 20
        else:
            x1 = self.x1
        return [[self.x0, x1], [self.y0, self.y1]]

class Datum:

    class States(Enum):
        DOWN = auto()   # Find the local low point
        UP = auto()     # Finding the local high point 
    
    DOWN = States.DOWN
    UP = States.UP

    def __init__(self, data):
        self.data = data
        self.datum = {'low': {'idx': [0], 'value': [data.low[0]]},  # The last one is not confirmed
                      'high': {'idx': [], 'value': []}}

        self.high_trend_line = TrendLine(0, 1e10, 0, 1e10)
        self.parallel_low_trend_line = TrendLine(0, 1e10, 0, 1e10)
        self.high_horizontal_line = TrendLine(0, 1e10, 0, 1e10)
        self.low_trend_line = TrendLine(0, 0, 0, 0)
        self.parallel_high_trend_line = TrendLine(0, 0, 0, 0)
        self.low_horizontal_line = TrendLine(0, 0, 0, 0)
        self.datum_lines = {'high_trend_line': [],
                            'parallel_low_trend_line': [],
                            'high_horizontal_line': [],
                            'low_trend_line': [],
                            'parallel_high_trend_line': [],
                            'low_horizontal_line': []}

        self.state: Datum.States = self.DOWN
        
        self.up_num = 0       # 上涨连续延续个数
        self.down_num = 0     # 下跌连续延续个数
        self.consolidate_num = 0       # 盘整个数

        self.last_low = data.low[0]
        self.last_high = data.high[0]

        self.i = 1

    def update_datum_line(self):
        if self.state == self.UP:
            # 低点已确定，更新低点水平线、低点趋势线、低点趋势平行线
            if len(self.datum['low']['value']) == 0:
                # 暂无低点
                return
            else:
                idx, value = self.datum['low']['idx'][-1], self.datum['low']['value'][-1]
                self.low_horizontal_line.update(idx, value, idx, value)

                # 只有低点趋势线第一次与之后的不同
                if len(self.datum['low']['value']) == 1:
                    # 目前只有一个低点
                    self.low_trend_line.update(idx, value, idx, value)
                else:
                    # 已经有多个低点
                    tmp = self.low_trend_line
                    tmp.update(tmp.x1, tmp.y1, idx, value)

                self.datum_lines['low_horizontal_line'].append(self.low_horizontal_line.values())
                # self.datum_lines['parallel_high_trend_line'].append(self.parallel_high_trend_line.values())
                self.datum_lines['low_trend_line'].append(self.low_trend_line.values())

                # 低点趋势平行线
                if len(self.datum['high']['value']) > 0:
                    tmp_idx = self.datum['high']['idx'][-1]
                    tmp_value = self.datum['high']['value'][-1]
                    self.parallel_low_trend_line.update(
                        tmp_idx, tmp_value, self.i, tmp_value + self.low_trend_line.k*(self.i - tmp_idx))
                    self.datum_lines['parallel_low_trend_line'].append(self.parallel_low_trend_line.values())

        else:
            # 高点已确定，更新高点水平线、高点趋势线、高点趋势平行线
            if len(self.datum['high']['value']) == 0:
                # 暂无高点
                return
            else:
                idx, value = self.datum['high']['idx'][-1], self.datum['high']['value'][-1]
                self.high_horizontal_line.update(idx, value, idx, value)

                # 只有高点趋势线第一次与之后的不同
                if len(self.datum['high']['value']) == 1:
                    # 目前只有一个高点
                    self.high_trend_line.update(idx, value, idx, value)
                else:
                    # 已经有多个高点
                    tmp = self.high_trend_line
                    tmp.update(tmp.x1, tmp.y1, idx, value)

                self.datum_lines['high_horizontal_line'].append(self.high_horizontal_line.values())
                # self.datum_lines['parallel_low_trend_line'].append(self.parallel_low_trend_line.values())
                self.datum_lines['high_trend_line'].append(self.high_trend_line.values())

                # 高点趋势平行线
                if len(self.datum['low']['value']) > 0:
                    tmp_idx = self.datum['low']['idx'][-1]
                    tmp_value = self.datum['low']['value'][-1]
                    self.parallel_high_trend_line.update(
                        tmp_idx, tmp_value, self.i, tmp_value + self.high_trend_line.k*(self.i-tmp_idx))
                    self.datum_lines['parallel_high_trend_line'].append(self.parallel_high_trend_line.values())

    def step(self):
        assert self.i < len(self.data)
        self._update_statistical_nums()

        if self._is_trend_continued():
            local_point = 'high' if self.state == self.UP else 'low'
            self.datum[local_point]['idx'][-1] = self.i
            self.datum[local_point]['value'][-1] = self.data[local_point][self.i]
            self._update_statistical_nums(clear=True)

        else:
            invert_trend_point_num = self.up_num if self.state == self.DOWN else self.down_num
            if invert_trend_point_num >= 2 or (invert_trend_point_num >= 1 and self.consolidate_num >= 2):
            # if invert_trend_point_num >= 3:
                # 如果趋势延续至少3次，或至少延续2次且有两次盘整
                self.update_datum_line()
                next_fake_point_type = 'low' if self.state == self.UP else 'high' 
                self.datum[next_fake_point_type]['idx'].append(self.i)
                self.datum[next_fake_point_type]['value'].append(self.data[next_fake_point_type][self.i])
                self._update_statistical_nums(clear=True)

                # Flip state
                self.state = self.UP if self.state == self.DOWN else self.DOWN
            # else:
                # do nothing

        self.last_low = self.data.low[self.i]
        self.last_high = self.data.high[self.i]

        self.i += 1

    def get_point(self, point, i=None):
        # h_p:          high_trend_line predict point
        # h_p_by_low:   high point predicted by parallel_low_trend_line
        # l_p:          low_trend_line predict point
        # l_p_by_high:  low point predicted by parallel_high_trend_line
        assert point in ['last_high', 'last_low', 'highs', 'lows'] or (
               point in ['h_p', 'h_p_by_low', 'l_p', 'l_p_by_high'] and i)

        if point in ['last_high', 'highs']:
            highs = self._get_confirmed_points('highs')
            if point == 'last_high':
                return highs[-1] if len(highs) > 0 else float('inf')
            else:
                return highs
        elif point in ['last_low', 'lows']:
            lows = self._get_confirmed_points('lows')
            if point == 'last_low':
                return lows[-1] if len(lows) > 0 else 0
            else:
                return lows
        elif point == 'h_p':
            return self.high_trend_line.predict(i)
        elif point == 'h_p_by_low':
            return self.parallel_low_trend_line.predict(i)
        elif point == 'l_p':
            return self.low_trend_line.predict(i)
        elif point == 'l_p_by_high':
            return self.parallel_high_trend_line.predict(i)
        else:
            raise ValueError

    def _update_statistical_nums(self, clear=False):
        if clear:
            self.down_num = 0
            self.up_num = 0
            self.consolidate_num = 0
        else:
            # Update up_num, down_num, consolidate_num
            low_lower = 1 if self.data.low[self.i] < self.last_low else 0
            high_lower = 1 if self.data.high[self.i] < self.last_high else 0

            if low_lower and high_lower:
                # down
                self.down_num += 1
                self.consolidate_num += self.up_num
                self.up_num = 0
            elif low_lower == 0 and high_lower == 0:
                # up
                self.up_num += 1
                self.consolidate_num += self.down_num
                self.down_num = 0
            else:
                self.consolidate_num += 1
                self.up_num = 0
                self.down_num = 0

    def _is_trend_continued(self) -> bool:
        # Found a lower low value when search local low point
        if self.state == self.DOWN and self.data['low'][self.i] < self.datum['low']['value'][-1]:
            return True
        # Found a higher high when search local high point
        elif self.state == self.UP and self.data['high'][self.i] > self.datum['high']['value'][-1]:
            return True
        else:
            return False

    def _get_confirmed_points(self, points_name) -> List:
        assert points_name in ['highs', 'lows']

        points = []
        if points_name == 'highs':
            if self.state == self.DOWN:
                highs = self.datum['high']['value'] if len(self.datum['high']['value']) > 0 else []
            else:
                highs = self.datum['high']['value'][:-1] if len(self.datum['high']['value']) > 1 else []
            points = highs
        elif points_name == 'lows':
            if self.state == self.UP:
                lows = self.datum['low']['value'][:] if len(self.datum['low']['value']) > 0 else []
            else:
                lows = self.datum['low']['value'][:-1] if len(self.datum['low']['value']) > 1 else []
            points = lows
        else:
            raise ValueError

        return points

def main():
    luna = Data('LUNABUSD', DataType.INTERVAL_1MINUTE).data.iloc[-1000:, :].reset_index(drop=True)
    luna['open_time'] = luna['open_time'].map(milliseconds_to_date)

    datum_class = Datum(luna)
    
    for i in range(1, len(luna)):
        datum_class.step()

    datum = datum_class.datum
    datum_lines = datum_class.datum_lines
    fig = PricePlot(luna['open'], luna['high'], luna['low'], luna['close'], luna['open_time'])
    fig.plot(plot_candle=True, plot_vol=False, datum=datum, datum_lines=None)

if __name__ == "__main__":
    from data import Data, DataType, milliseconds_to_date
    main()
    print("Finished")