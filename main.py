import time
import os
from adapter import Adaptor, AdaptorBinance, AdaptorSimulator
from new_policy import Policy, PolicyBreakThrough, PolicyBreakThrough2
from base_types import DataType, DataElements
from data import Data
from plot import PricePlot
from utils import milliseconds_to_date
from logger import Logger
import sys
from account_state import AccountState
            

def main_loop(adaptor: Adaptor, policy: Policy, log_en=False):
    state = AccountState(adaptor, analyze_en=policy.analyze_en, log_en=log_en)

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

            if is_trade == False and state.can_sell():
                is_trade = policy.try_to_sell(adaptor)

            # If trade, update             
            if is_trade:
                state.update()

            if adaptor.is_next_step():
                break

            # Wait a new price
            new_price = adaptor.get_price()
            while new_price == price:
                time.sleep(0.2)
                new_price = adaptor.get_price()
            price = new_price
        
        last_timestamp = adaptor.get_timestamp()-60000
        
        if is_trade:
            state.update_analyzed_info(last_timestamp)

        policy.update(high = adaptor.get_latest_kline_value(DataElements.HIGH),
                    low = adaptor.get_latest_kline_value(DataElements.LOW),
                    open = adaptor.get_latest_kline_value(DataElements.OPEN),
                    close = adaptor.get_latest_kline_value(DataElements.CLOSE),
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
    
    earn_point = state.get_earn_points(data)

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
    policy = PolicyBreakThrough(adaptor.get_timestamp(), log_en=log_en, analyze_en=analyze_en, policy_private_log=policy_private_log)
    for i in range(data.len()):
        policy.update(high = data.get_value(DataElements.HIGH, i),
                      low = data.get_value(DataElements.LOW, i),
                      open = data.get_value(DataElements.OPEN, i),
                      close = data.get_value(DataElements.CLOSE, i),
                      timestamp = int(data.get_value(DataElements.OPEN_TIME, i)))

    while True:
        try:
            main_loop(adaptor, policy, log_en)
        except KeyboardInterrupt:
            break
        except Exception as ex:
            # traceback.print_exc()
            adaptor.clear_open_orders()
            print(ex)
            time.sleep(30)
            if data.update(end_str="1 minute ago UTC+8"):
                policy.update(high = data.get_value(DataElements.HIGH, -1),
                            low = data.get_value(DataElements.LOW, -1),
                            open = data.get_value(DataElements.OPEN, -1),
                            close = data.get_value(DataElements.CLOSE, -1),
                            timestamp = int(data.get_value(DataElements.OPEN_TIME, -1)))


def simulated_trade():
    usd_name = 'BUSD'
    token_name='LUNA2'
    # token_name = 'BTC'

    log_en = False
    analyze_en = True
    save_info = False
    exp_name = 'threshold_30_k0.4_p2_fixErrorPoint_newStruct'

    print('Loading data')
    symbol = token_name+usd_name
    data = Data(symbol, DataType.INTERVAL_1MINUTE, 
                # start_str="2022-06-08 18:48 UTC+8", end_str="2022/06/09 8:27 UTC+8", is_futures=True)
                
                # Found a Mistake Top 
                # start_str="2022-06-09 0:15 UTC+8", end_str="2022/06/09 4:40 UTC+8", is_futures=True)
                
                # Small cycle is embedded in Big cycle
                # start_str="2022-06-15 15:15 UTC+8", end_str="2022/06/16 0:00 UTC+8", is_futures=True)
                is_futures=True)
    print('Loading data finished')

    adaptor = AdaptorSimulator(usd_name=usd_name, token_name=token_name, init_balance=1000000, 
                               leverage=1, data=data, fee=0.00038, log_en=log_en)
    # policy = PolicyBreakThrough(adaptor.get_timestamp(), log_en=log_en, analyze_en=analyze_en)
    policy = PolicyBreakThrough2(adaptor.get_timestamp(), log_en=log_en, analyze_en=analyze_en)

    start = time.time()
    state = main_loop(adaptor, policy, log_en)
    end = time.time()
    print('Main loop execution time is {:.3f}s'.format(end - start))

    if save_info and analyze_en:
        folder = '.\\log\\{}'.format(exp_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        policy.save(folder, symbol, data.start_time(), data.end_time())
        state.save_earn_points(data, folder, symbol, data.start_time(), data.end_time())
    
    try:
        final_log(data, policy, state)
    except Exception as ex:
        print(ex)

    if analyze_en:
        plot(data, policy, state)


if __name__ == "__main__":
    real = False
    if real:
        # Log to file
        file_path = '.\\log\\log_{}.txt'.format(milliseconds_to_date(int(time.time())*1000).replace(':', '-'))
        print('Log to file: {}'.format(file_path))
        sys.stdout = Logger(file_path)
        real_trade()
    else:
        simulated_trade()
    print("Finished")