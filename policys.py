from typing import Dict, Optional, List
from abc import ABC, abstractmethod
import pandas as pd
from datum import Datum
from plot import PricePlot
from base_types import IdxValue, OptPoints, OptState

# Trade simulator which should have same interface with the Binance trade wrapper
# class TradeSimulator():


# Base class of policy
class PolicyBase(ABC):

    def __init__(self, open, high, low, close, open_time, fee, log_enable=False):
        self.data = pd.DataFrame()
        self.data['open'] = open
        self.data['high'] = high
        self.data['low'] = low
        self.data['close'] = close
        self.data['timestamp'] = open_time
        self.data = self.data.reset_index(drop=True)

        self.fee = fee
        self.log_enable = log_enable

        self.datum_class = Datum(self.data)
        self.opt_point = OptPoints()
        self.earn_curve = IdxValue(idx=[0], value=[1])
        self.stock_nums = 1
        self.buy_state = []
        self.sell_state = []
        self.buy_reason = None
        self.sell_reason = None
        self.buy_sell_state_init()

    @abstractmethod
    def buy_sell_state_init(self):
        # buy_state 和 sell_state 初始化
        raise NotImplementedError

    def buy(self, i, datum_point, strategy, log=''):
        open = self.data['open'].iloc[i]
        low = self.data['low'].iloc[i]
        high = self.data['high'].iloc[i]
        
        assert (strategy == 'up' and datum_point <= high) or \
            (strategy == 'down' and datum_point >= low), "datum_point not correct"
        
        # Calculate price
        opt_fun = min if strategy == 'down' else max
        price = opt_fun(datum_point, open)

        # Stock nums to buy
        self.stock_nums = self.earn_curve.value[-1] * self.data['close'][0] / price * (1 - self.fee)
        
        # Calculate earn of current timestamp
        self.opt_point.buy.add(i, price)
        earn = (self.data['close'].iloc[i] - price) * self.stock_nums / self.data['close'].iloc[0] \
               + self.earn_curve.value[-1]
        
        if self.log_enable:
            print('\n{}: buy, price = {:.7f}'.format(self.data['timestamp'][i], price))
            print(log)
        
        return 'full', earn

    def sell(self, i, datum_point, strategy, log=''):
        open = self.data['open'].iloc[i]
        low = self.data['low'].iloc[i]
        high = self.data['high'].iloc[i]

        assert (strategy == 'up' and datum_point <= high) or \
            (strategy == 'down' and datum_point >= low), "datum_point not correct"
        
        # Calculate price
        opt_fun = min if strategy == 'down' else max
        price = opt_fun(open, datum_point)
        
        # Calculate earn of current timestamp
        self.opt_point.sell.add(i, price)
        get_money = price * self.stock_nums * (1 - self.fee)
        earn = (get_money - (self.data['close'].iloc[i - 1] * self.stock_nums)) / \
                self.data['close'].iloc[0] + self.earn_curve.value[-1]

        success = True if self.opt_point.sell.value[-1] > self.opt_point.buy.value[-1] else False
        if self.log_enable:
            print('{}: sell, price = {:.7f}'.format(self.data['timestamp'][i], price))
            print(log)
            print('单笔交易盈利' if success else '单笔交易亏损')
        return 'empty', earn, success

    @abstractmethod
    def buy_strategy(self, i):
        # 买入：   返回 买入点位 上穿('up')/下穿('down') buy_reason log
        # 不买入：  返回 None
        raise NotImplementedError

    @abstractmethod
    def sell_strategy(self, i):
        # 卖出：   返回 卖出点位 上穿('up')/下穿('down')
        # 不卖出：  返回 None
        raise NotImplementedError

    def calc_opt_point(self):
        hold_state = 'empty'  # full 表示满仓，empty 表示空仓
        init_price = self.data['close'].iloc[0]
        self.log_enable = False
        for i in range(1, len(self.data)):
            if self.log_enable == False and (i+1) % 1000 == 0:
                print('Idx: {} / {}'.format(i+1, len(self.data)), end='\r')

            self.datum_class.step()
            last_price = self.data['close'].iloc[i-1]
            curr_price = self.data['close'].iloc[i]
            
            if hold_state == 'full':
                # Whether sell
                sell_price, up_down, sell_reason, log = self.sell_strategy(i)
                
                if sell_price:
                    # If sell
                    hold_state, earn, success = self.sell(i, sell_price, up_down, log)
                    # success means earned in this trading
                    self.buy_state[self.buy_reason]['num'][sell_reason] += 1
                    self.buy_state[self.buy_reason]['successful_num'][sell_reason] += 1 if success else 0
                    self.sell_state[sell_reason]['num'][self.buy_reason] += 1
                    self.sell_state[sell_reason]['successful_num'][self.buy_reason] += 1 if success else 0
                else:
                    # If keep holding, calculate earn
                    earn = (curr_price - last_price) * self.stock_nums / init_price \
                           + self.earn_curve.value[-1]

                self.earn_curve.add(i, earn)

            else:
                # Whether buy
                buy_price, up_down, buy_reason, log = self.buy_strategy(i)
                if buy_price:
                    # buy
                    self.buy_reason = buy_reason
                    hold_state, earn = self.buy(i, buy_price, up_down, log)
                    self.earn_curve.add(i-1, self.earn_curve.value[-1])
                    self.earn_curve.add(i, earn)

        self.print_log()

    def datum_point(self, name, i=None):
        # 查询单个基准点或多个，单个name为字符，多个为字符的list
        if isinstance(name, str):
            name = [name]
        rt = [self.datum_class.get_point(n, i) for n in name]
        return rt if len(rt) > 1 else rt[0]

    def print_log(self):
        print()
        print()
        print('First day is {}, finally day = {}'.format(self.data['timestamp'][0], self.data['timestamp'].iloc[-1]))
        print('Init price = {}, current price = {}'.format(self.data['close'].iloc[0], self.data['close'].iloc[-1]))
        print('Earn = {:.7f}%, base line = {:.7f}%, '.format(
            self.earn_curve.value[-1] * 100 - 100,
            (self.data['high'].iloc[-1] - self.data['low'][0]) / self.data['close'].iloc[0] * 100))

        n_buy = len(self.opt_point.buy.value)
        n_sell = len(self.opt_point.sell.value)
        print('n_buy = {}, n_sell = {}'.format(n_buy, n_sell))
        print('Sum of commission = {:.7f}%'.format((n_buy + n_sell) * 0.025))
        print('Successful rate = {:.7f}%'.format(
            sum([self.opt_point.sell.value[i] > self.opt_point.buy.value[i] for i in range(n_sell)])
            / (n_sell * 100 + 1e-5)))

        print()
        for buy_reason in self.buy_state:
            print('买入原因：{}, 个数：{}, 成功个数：{}, 成功率:{}, 总成功率：{:.7f}'.format(
                buy_reason['name'], buy_reason['num'], buy_reason['successful_num'],
                [buy_reason['successful_num'][i] / (buy_reason['num'][i]+1e-5) for i in range(len(self.sell_state))],
                sum(buy_reason['successful_num']) / (sum(buy_reason['num'])+1e-5)))

        for sell_reason in self.sell_state:
            print('卖出原因：{}, 个数：{}, 成功个数：{}, 成功率:{}, 总成功率：{:.7f}'.format(
                sell_reason['name'], sell_reason['num'], sell_reason['successful_num'],
                [sell_reason['successful_num'][i] / (sell_reason['num'][i]+1e-5) for i in range(len(self.buy_state))],
                sum(sell_reason['successful_num']) / (sum(sell_reason['num'])+1e-5)))


# 策略1，突破上一高点买，突破上一低点卖
class Policy1(PolicyBase):
    name = 'Policy 1'

    def buy_sell_state_init(self):
        # buy_state 和 sell_state 初始化
        self.buy_state = [{'name': '上穿上一高点买入', 'num': [0], 'successful_num': [0]}]
        self.sell_state = [{'name': '下穿上一低点卖出', 'num': [0], 'successful_num': [0]}]

    def buy_strategy(self, i):
        # 买入：   返回 买入点位 上穿('up')/下穿('down') buy_reason log
        # 不买入：  返回 None, None, None, None
        last_high = self.datum_point('last_high')
        if self.data['high'][i] > last_high:
            return last_high, 'up', 0, '上穿上一高点，买入'
        else:
            return None, None, None, None

    def sell_strategy(self, i):
        # 卖出：   返回 卖出点位 上穿('up')/下穿('down') buy_reason(0, 1, 2 ...) log
        # 不卖出：  返回 None
        last_low = self.datum_point('last_low')
        if self.data['low'][i] < last_low:
            return last_low, 'down', 0, '下穿上以低点，卖出'
        else:
            return None, None, None, None


def main():
    # luna = Data('BTCUSDT', DataType.INTERVAL_1MINUTE, "2022/3/1 UTC+8", "2020/5/1 UTC+8").data
    # luna = Data('BTCUSDT', DataType.INTERVAL_1MINUTE).data.iloc[-50000:, :].reset_index(drop=True)
    # luna = Data('BTCBUSD', DataType.INTERVAL_1MINUTE).data.iloc[-100000:, :].reset_index(drop=True)
    luna = Data('BTCBUSD', DataType.INTERVAL_1MINUTE, is_futures=True).data.iloc[-1000:, :].reset_index(drop=True)
    # luna = Data('LUNABUSD', DataType.INTERVAL_1MINUTE).data
    # luna = Data('LUNCBUSD', DataType.INTERVAL_1MINUTE).data
    luna['open_time'] = luna['open_time'].map(milliseconds_to_date)

    policy = Policy1(luna['open'], luna['high'], luna['low'], luna['close'], luna['open_time'], 0.0002, log_enable=False)
    policy.calc_opt_point()
    datum = policy.datum_class.datum
    datum_lines = policy.datum_class.datum_lines
    opt_point = policy.opt_point
    earn_curve = policy.earn_curve
    fig = PricePlot(luna['open'], luna['high'], luna['low'], luna['close'], luna['open_time'])
    fig.plot(plot_candle=len(luna)<=1100, plot_vol=False, datum=datum, datum_lines=None,
                    opt_point=opt_point, earn_point=earn_curve)

if __name__ == "__main__":
    from data import Data, DataType, milliseconds_to_date
    main()
    print("Finished")