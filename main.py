import time
from adapter import Adaptor, AdaptorBinance, AdaptorSimulator
from new_policy import Policy, PolicyBreakThrough
from base_types import DataType, DataElements, IdxValue
from data import Data
from plot import PricePlot
from utils import milliseconds_to_date

class AccountState:

    def __init__(self, adaptor: Adaptor, analyze_en: bool=True) -> None:
        self.balance: float = adaptor.balance()
        self.init_balance: float = self.balance
        self.pos_amount: float = adaptor.pos_amount()
        self.analyze_en: bool = analyze_en
        self.adaptor: Adaptor = adaptor

        if self.analyze_en:
            timestamp = adaptor.get_timestamp()
            self.update_times = [timestamp]
            self.balance_history = [self.balance]
            self.pos_amount_history = [self.pos_amount]
    
    def can_buy(self) -> bool:
        assert self.pos_amount >= 0, "Currently not support SHORT side"
        if self.pos_amount == 0:
            return True
        else:
            return False
    
    def can_sell(self) -> bool:
        if self.pos_amount > 0:
            return True
        else:
            return False
    
    def update_analyzed_info(self, timestamp: int) -> None:
        # Save analyze info
        if self.analyze_en:
            self.update_times.append(timestamp)
            self.balance_history.append(self.balance)
            self.pos_amount_history.append(self.pos_amount)

    def update(self) -> None:
        self.balance: float = self.adaptor.balance()
        self.pos_amount: float = self.adaptor.pos_amount()
    
    def earn_rate(self, price: float) -> float:
        # 1 represent 100% means no earn
        return self.earn(price) / self.init_balance

    def earn(self, price: float) -> float:
        if self.pos_amount > 0:
            buy_price = self.adaptor.entry_price()
            earn = (price - buy_price) * self.pos_amount

            leverage = self.adaptor.get_leverage()
            buy_balance = buy_price * self.pos_amount / leverage

            balance = self.balance + buy_balance + earn
        else:
            balance = self.balance

        total_earn = balance - self.init_balance
        return total_earn

    def get_earn_points(self, data: Data, buy_points: IdxValue) -> IdxValue:
        # points idx must be timestamp
        earn_points = IdxValue()
        if self.analyze_en:
            assert len(self.update_times) == len(self.balance_history) == len(self.pos_amount_history)
            
            # Convert IdxValue to dict[timestamp -> value]
            time_to_buy_price = dict(zip(buy_points.idx, buy_points.value))

            # Get init eran_points' idx and the value before first trade
            earn_points.idx = data.data.index.to_list()
            earn_points.value = [0.0] * data.len()
            init_balance = self.balance_history[0]
            i = 0
            for j in range(len(self.update_times)):
                if self.pos_amount_history[j] > 0:
                    # It is a buy operation
                    
                    # 1. Calculate earn of current timestamp
                    if j > 0:
                        close = data.get_value(DataElements.CLOSE, i)
                        buy_price = time_to_buy_price[self.update_times[j]]
                        pos_amount = self.pos_amount_history[j]
                        earn = (close - buy_price) * pos_amount + self.balance_history[j]
                        earn_rate = earn / init_balance
                        earn_points.value[i] = earn_points.value[i-1] + earn_rate
                    else:
                        earn_points.value[i] = 1
                    i += 1

                    # 2. Calculate earn until next update
                    close = data.get_value(DataElements.CLOSE, i-1)
                    while (j+1 < len(self.update_times) and \
                            self.update_times[j+1] > data.get_value(DataElements.OPEN_TIME, i)) or \
                          (j+1 >= len(self.update_times) and i < data.len()):

                        last_close = close
                        close = data.get_value(DataElements.CLOSE, i)
                        earn = (close - last_close) * self.pos_amount_history[j]
                        earn_rate = earn / init_balance
                        earn_points.value[i] = earn_points.value[i-1] + earn_rate
                        i += 1

                else:
                    # It is a sell operation
                    
                    # 1. Calculate earn of current timestamp
                    if j > 0:
                        earn_points.value[i] = self.balance_history[j] / init_balance
                    else:
                        earn_points.value[i] = 1
                    i += 1

                    # 2. Calculate earn until next update
                    while (j+1 < len(self.update_times) and \
                            self.update_times[j+1] > data.get_value(DataElements.OPEN_TIME, i)) or \
                          (j+1 >= len(self.update_times) and i < data.len()):
                        
                        earn_points.value[i] = earn_points.value[i-1]
                        i += 1

                # pos_amount is 0
            # loop for each states
        # if self.analyze_en:
 
        return earn_points
            


def main_loop(adaptor: Adaptor, policy: Policy, log_en=False):
    state = AccountState(adaptor, analyze_en=policy.analyze_en)

    i = 0
    while True:
        if log_en == False and (i+1) % 1000 == 0:
            print('Idx: {}'.format(i+1), end='\r')

        # For each step
        price = adaptor.get_price()
        is_trade = False
        while True:
            # For dynamic price in one step
            if state.can_buy():
                is_trade = policy.try_to_buy(adaptor)
            elif state.can_sell():
                is_trade = policy.try_to_sell(adaptor)
            else:
                is_trade = False

            # If trade, update             
            if is_trade:
                state.update()

            if adaptor.is_next_step():
                break

            # Wait a new price
            new_price = adaptor.get_price()
            while new_price == price:
                time.sleep(0.5)
                new_price = adaptor.get_price()
            price = new_price
        
        last_timestamp = adaptor.get_timestamp()-60000
        
        if is_trade:
            state.update_analyzed_info(last_timestamp)

        policy.update(high = adaptor.get_latest_kline_value(DataElements.HIGH),
                      low = adaptor.get_latest_kline_value(DataElements.LOW),
                      timestamp = last_timestamp)
        
        # Used for simulator
        if adaptor.is_finished():
            break
        
        i += 1

    return state


def plot(data: Data, policy: PolicyBreakThrough, state: AccountState):
    open, high, low, close, open_time = data.get_columns([
        DataElements.OPEN, DataElements.HIGH, DataElements.LOW, DataElements.CLOSE, DataElements.OPEN_TIME])    
    open_time = open_time.map(milliseconds_to_date)
    
    fig = PricePlot(open, high, low, close, open_time)
    
    buy_points = policy.get_points(policy.PointsType.ACTUAL_BUY)
    sell_points = policy.get_points(policy.PointsType.ACTUAL_SELL)
    
    # earn_points need the idx to be timestamp
    earn_point=state.get_earn_points(data, buy_points)

    tops = policy.tops
    bottoms = policy.bottoms
    for point in [buy_points, sell_points, tops, bottoms]:
        point.idx = data.time_list_to_idx(point.idx)

    points = [
        PricePlot.Points(idx=buy_points.idx, value=buy_points.value, s=90, c='r', label='buy'),
        PricePlot.Points(idx=sell_points.idx, value=sell_points.value, s=90, c='g', label='sell'),
        PricePlot.Points(idx=tops.idx, value=tops.value, s=30, c='b', label='top'),
        PricePlot.Points(idx=bottoms.idx, value=bottoms.value, s=30, c='y', label='bottoms'),
    ]
    fig.plot(plot_candle=data.len()<=1100, points=points, earn_point=earn_point)


def final_log(data: Data, policy: Policy, state: AccountState):
    print()
    print()
    print('Time begin from {} to {}'.format(data.start_time_str(), data.end_time_str()))
    
    init_price = data.get_value(DataElements.CLOSE, 0)
    current_price = data.get_value(DataElements.CLOSE, -1)
    print('Init price = {}, current price = {}'.format(
        init_price, current_price))
    print('Earn = {:.3f}%, base line = {:.3f}%, '.format(
        state.earn_rate(current_price) * 100,
        (current_price - init_price) / init_price * 100
    ))

    policy.log_analyzed_info()


def real_trade():
    usd_name = 'BUSD'
    token_name='LUNA2'
    log_en = True
    analyze_en = True

    policy_private_log = True

    # Updata data to latest
    data = Data(token_name+usd_name, DataType.INTERVAL_1MINUTE, is_futures=True)
    adaptor = AdaptorBinance(usd_name=usd_name, token_name=token_name, data=data, log_en=log_en)
    data.set_client(adaptor.client)
    data.update(end_str="1 minute ago UTC+8")
    data.replace_data_with_range(num=20)
    print('Data start with {}, end with {}'.format(data.start_time_str(), data.end_time_str()))

    # Update policy
    policy = PolicyBreakThrough(log_en=log_en, analyze_en=analyze_en, policy_private_log=policy_private_log)
    for i in range(data.len()):
        policy.update(high = data.get_value(DataElements.HIGH, i),
                      low = data.get_value(DataElements.LOW, i),
                      timestamp = int(data.get_value(DataElements.OPEN_TIME, i)))

    main_loop(adaptor, policy, log_en)


def simulated_trade():
    usd_name = 'BUSD'
    token_name='LUNA2'
    # token_name = 'BTC'
    log_en = False
    analyze_en = True

    print('Loading data')
    # data = Data(token_name+usd_name, DataType.INTERVAL_1MINUTE, start_str="2022/06/02 21:00 UTC+8", is_futures=True)
    data = Data(token_name+usd_name, DataType.INTERVAL_1MINUTE, num=100, is_futures=True)
    print('Loading data finished')

    adaptor = AdaptorSimulator(usd_name=usd_name, token_name=token_name, init_balance=1000000, 
                               leverage=1, data=data, fee=0.0000, log_en=log_en)
    policy = PolicyBreakThrough(log_en=log_en, analyze_en=analyze_en)

    start = time.time()
    state = main_loop(adaptor, policy, log_en)
    end = time.time()
    print('Main loop execution time is {:.3f}s'.format(end - start))

    final_log(data, policy, state)
    if analyze_en:
        plot(data, policy, state)


if __name__ == "__main__":
    real = False
    if real:
        real_trade()
    else:
        simulated_trade()
    print("Finished")