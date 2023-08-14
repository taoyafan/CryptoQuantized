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
            
def main_loop(adaptor: AdaptorBinance, log_en=False):
    marked_price = 0
    last_min = 0
    last_second = 0
    enter_price = 0
    exit_price = 0

    while True:
        time_ms = adaptor.get_timestamp()
        price = adaptor.get_price()
        ms = time_ms % 1000 // 10

        time_s = time_ms // 1000
        second = time_s % 60

        time_min = time_s // 60
        min = time_min % 60

        time_hour = time_min // 60
        hour = (time_hour + 8) % 24
        
        if last_min != min:
            print(f"\n--- New minute: {hour :02d}:{min :02d}:")

        if second == 58:
            print(f"{second :02d}.{ms :02d}: {price :.2f}", end="\t\t")
            if marked_price == 0:
                marked_price = price # Set the first time
            
        if second == 59:
            if last_second != 59:
                print()

            print(f"{second :02d}.{ms :02d}: {price :.2f}", end="\t\t")
            
            assert marked_price != 0
            if price > marked_price + 3:
                enter_price = marked_price + 3
            elif price < marked_price - 3:
                enter_price = marked_price - 3

        if second == 0 and last_second != 0:
            if enter_price != 0:
                print(f"Entered, enter price: {price :.2f}", end="\t\t")
                exit_price = marked_price
            else:
                time.sleep(55)
        
            marked_price = 0    # Reset marked price

        if exit_price != 0 and (second >= 0 or second <= 8):
            if last_second != second:
                print()
            
            print(f"{second :02d}.{ms :02d}: {price :.2f}", end="\t\t")
            if ((exit_price < enter_price and price <= exit_price) or 
                (exit_price > enter_price and price >= exit_price)
            ):
                # Short exit:
                print(f"\nShort exit")
                exit_price = 0
                enter_price = 0
                time.sleep(48)
        
        # Force exit at second 9
        if exit_price != 0 and last_second != second and second == 9:
            earn = 0
            if exit_price < enter_price:
                earn = enter_price - price
            else:
                earn = price - enter_price
            
            print(f"\nForce exit at 9, price: {price :.2f}, earn: {earn :.2f}")
            exit_price = 0
            enter_price = 0
            time.sleep(45)

        last_min = min
        last_second = second
        
        time.sleep(0.1)

def real_trade():
    usd_name = 'TUSD'
    token_name='BTC'
    is_futures=False
    log_en = True

    # Updata data to latest
    data = Data(token_name+usd_name, DataType.INTERVAL_1MINUTE, num=100, is_futures=is_futures)
    adaptor = AdaptorBinance(usd_name=usd_name, token_name=token_name, data=data, log_en=log_en, is_futures=is_futures)

    error_occured = False
    while True:
        # try:
        main_loop(adaptor=adaptor, log_en=log_en)

        # except KeyboardInterrupt:
        #     break
        # except Exception as ex:
        #     # traceback.print_exc()
        #     logging.exception(ex)
        #     print('Return to main, Retry')
        #     error_occured = True

        # if error_occured:
        #     error_occured = False
        #     adaptor.reset()
        #     if adaptor.pos_amount() > 0:
        #         adaptor._order_market(OrderSide.SELL, adaptor.pos_amount())

        #     print('Clear all open order')
        #     time.sleep(30)

if __name__ == "__main__":
    # Log to file
    log_to_file = True

    if log_to_file:
        path = os.getcwd()
        file_path = os.path.join(path,'log')
        file_path = os.path.join(file_path,'log_{}.txt'.format(
            milliseconds_to_date(int(time.time())*1000).replace(':', '-').replace(' ', '_')))
        print('Log to file: {}'.format(file_path))
        sys.stdout = Logger(file_path)
        sys.stderr = sys.stdout
        
    real_trade()
    print("Finished")