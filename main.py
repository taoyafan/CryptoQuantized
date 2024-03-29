import time
import os
from adapter import Adaptor, AdaptorBinance, AdaptorSimulator
from new_policy import *
from policy_MA import PolicyMA
from base_types import DataType, DataElements
from data import Data
from plot import PricePlot
from utils import milliseconds_to_date
from logger import Logger
import sys
from account_state import AccountState
import logging
            

def main_loop(state: AccountState, adaptor: Adaptor, policy: Policy, log_en=False):

    i = 0
    while True:
        if log_en == False and (i+1) % 1000 == 0:
            print('Idx: {}'.format(i+1), end='\r')

        # For each step
        price = adaptor.get_price()
        new_step = True
        while True:
            # For dynamic price in one step
            policy.try_to_buy(new_step)
            policy.try_to_sell(new_step)
            
            state.update()
            
            if adaptor.is_next_step():
                break

            # Wait a new price
            new_price = adaptor.get_price()
            while new_price == price:
                time.sleep(0.1) # Update per 100 ms
                new_price = adaptor.get_price()
            price = new_price
            new_step = False

        last_timestamp = int(adaptor.get_latest_kline_value(DataElements.OPEN_TIME))
        timestamp = last_timestamp + 60000
        state.update_each_time_step(timestamp)
        
        policy.update(
                    high = adaptor.get_latest_kline_value(DataElements.HIGH),
                    low = adaptor.get_latest_kline_value(DataElements.LOW),
                    close = adaptor.get_latest_kline_value(DataElements.CLOSE),
                    volume = adaptor.get_latest_kline_value(DataElements.QUOTE_ASSERT_VOLUME),
                    timestamp = last_timestamp)
        
        # Used for simulator
        if adaptor.is_finished():
            break
        
        i += 1

    return state


def plot(data: Data, policy: Policy, state: AccountState):
    open, high, low, close, open_time = data.get_columns([
        DataElements.OPEN, DataElements.HIGH, DataElements.LOW, DataElements.CLOSE, DataElements.OPEN_TIME])    
    open_time = open_time.map(milliseconds_to_date)
    
    fig = PricePlot(open, high, low, close, open_time)
    earn_point = state.get_earn_points(data)

    fig.plot(plot_candle = data.len()<=1100, 
             points      = policy.get_plot_points(data), 
             earn_point  = earn_point,
             atr60       = policy.get_atr60(),
             mas= [3, 10, 60])


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
    usd_name = 'FDUSD'
    token_name='ETH'
    is_futures=False
    log_en = True
    analyze_en = False
    policy_private_log = True

    # Updata data to latest
    data = Data(token_name+usd_name, DataType.INTERVAL_1MINUTE, is_futures=is_futures)
    adaptor = AdaptorBinance(usd_name=usd_name, token_name=token_name, data=data, log_en=log_en, is_futures=is_futures)
    data.set_client(adaptor.client)
    data.update(end_str="1 minute ago UTC+8")
    data.replace_data_with_range(num=300)
    print('Data start with {}, end with {}'.format(data.start_time_str(), data.end_time_str()))

    state = AccountState(adaptor, analyze_en=analyze_en, log_en=log_en)

    # Update policy
    fee = 0.00001
    timestamp = int(data.get_value(DataElements.OPEN_TIME, 0))

    k_same_points_delta = 0
    k_other_points_delta = 0
    k_from_latest_point = 0
    search_to_now = False

    policy = PolicySwing(
        state,
        timestamp, 
        log_en = log_en, 
        analyze_en = analyze_en, 
        policy_private_log = policy_private_log,

        k_same_points_delta = k_same_points_delta,
        k_other_points_delta = k_other_points_delta,
        k_from_latest_point = k_from_latest_point,
        search_to_now = search_to_now,
        
        fee = fee)

    def update_policy(i):
        policy.update(high = data.get_value(DataElements.HIGH, i),
                      low = data.get_value(DataElements.LOW, i),
                      close = data.get_value(DataElements.CLOSE, i),
                      volume = data.get_value(DataElements.VOLUME, i),
                      timestamp = int(data.get_value(DataElements.OPEN_TIME, i)))

    for i in range(data.len()):
        update_policy(i)

    error_occured = False
    
    while True:
        try:
            if error_occured:
                error_occured = False
                adaptor.reset()
                if adaptor.pos_amount() > 0:
                    adaptor._order_market(OrderSide.SELL, adaptor.pos_amount())
                state.reset()
                policy.reset()

                print('Clear all open order')
                time.sleep(30)
                if data.update(end_str="1 minute ago UTC+8"):
                    update_policy(-1)

            main_loop(state, adaptor, policy, log_en)
        except KeyboardInterrupt:
            break
        except Exception as ex:
            # traceback.print_exc()
            logging.exception(ex)
            print('Return to main, Retry')
            error_occured = True


def simulated_trade():
    # usd_name = 'BUSD'
    # usd_name = 'TUSD'
    usd_name = 'FDUSD'
    # usd_name = 'USDT'
    # token_name='LUNA2'
    # token_name='1000LUNC'
    # token_name='DOGE'
    # token_name='GMT'
    token_name = 'BTC'
    # token_name = 'ETH'
    # token_name = 'SOL'
    is_futures = False

    log_en = False
    analyze_en = True
    save_info = False
    
    k_same_points_delta = 0
    k_other_points_delta = 0
    k_from_latest_point = 0
    search_to_now = False

    # ksol: k_same_points_delta, k_other_points_delta, k_from_latest_point
    # SearchtoNow: search to max (currunt time, threshold) when update policy
    # mfp: move fake point to the correct pos
    # frontEn: k_other_points_delta works as the front min delta time
    # exp_name = 'ksol_{}_{}_{}{}'.format(k_same_points_delta, k_other_points_delta, 
    #                                     k_from_latest_point, '_SearchToNow' if search_to_now else '')
    exp_name = "SellAfter2Step-TH2"
    print('Exp name: {}'.format(exp_name))
    print('Loading data')
    symbol = token_name+usd_name
    mtd = milliseconds_to_date
    data = Data(symbol, DataType.INTERVAL_1MINUTE, 
                # Test
                # start_str="2023-07-29 06:04:59.999000", is_futures=is_futures) # Note no UTC+8
                # end_str='2023-06-14 11:24:00 UTC+8', num=100000, is_futures=is_futures)
                # start_str='2023-08-26 00:00:00 UTC+8', num=1000, is_futures=is_futures)
                num=100000, is_futures=is_futures)
                # start_str='2023-08-31 18:00:00 UTC+8', end_str='2023-08-31 20:00:00 UTC+8', is_futures=is_futures)
                # start_str='2023-09-09 00:01:00 UTC+8', num=100000, is_futures=is_futures)
                # start_str=mtd(1682655420000) + ' UTC+8',  end_str=mtd(1692364319999) + ' UTC+8', is_futures=is_futures)
                # start_str=milliseconds_to_date(1682655420000) + ' UTC+8', num=1000, is_futures=is_futures)

    print('Loading data finished')

    fee = 0.00002
    interest = 0.00001
    adaptor = AdaptorSimulator(usd_name=usd_name, token_name=token_name, init_balance=1000000, 
                               leverage=1, data=data, fee=fee, interest=interest, log_en=log_en)

    state = AccountState(adaptor, analyze_en=analyze_en, log_en=log_en)

    # policy = PolicyBreakThrough(state, adaptor.get_timestamp(), log_en=log_en, analyze_en=analyze_en)
    # policy = PolicyBreakThrough( 
    policy = PolicySwing(
    # policy = PolicyBreakWithMa300Low(
    # policy = PolicyDelayAfterBreakThrough(
        state,
        adaptor.get_timestamp(), 
        log_en = log_en, 
        analyze_en = analyze_en, 
        policy_private_log = True,
        
        k_same_points_delta = k_same_points_delta,
        k_other_points_delta = k_other_points_delta,
        k_from_latest_point = k_from_latest_point,
        search_to_now = search_to_now,
        fee = fee)

    # policy = PolicyMA(
    #     state      = state,
    #     level_fast = 5,
    #     level_slow = 15,
    #     log_en     = log_en, 
    #     analyze_en = analyze_en)

    start = time.time()
    state = main_loop(state, adaptor, policy, log_en)
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
        path = os.getcwd()
        file_path = os.path.join(path,'log')
        file_path = os.path.join(file_path,'log_{}.txt'.format(
            milliseconds_to_date(int(time.time())*1000).replace(':', '-').replace(' ', '_')))
        print('Log to file: {}'.format(file_path))
        sys.stdout = Logger(file_path)
        sys.stderr = sys.stdout
        real_trade()
    else:
        simulated_trade()
    print("Finished")