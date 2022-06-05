from abc import ABC, abstractmethod
from typing import Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd
import time
from enum import Enum
from utils import milliseconds_to_date

from api_key import API_KEY, SECRET_KEY
from base_types import PolicyToAdaptor, DataElements
from data import Data

class Adaptor(ABC):

    token_min_pos_table = {
        'BTC': 0.001,
        'LUNA2': 1,
    }

    def __init__(self, usd_name, token_name, data:Data, log_en):
        self.usd_name = usd_name
        self.token_name = token_name
        self.symbol = token_name + usd_name
        self.data = data

        assert token_name in self.token_min_pos_table, 'Not support token'
        self.token_min_pos = self.token_min_pos_table[token_name]
        self.log_en = log_en

    @abstractmethod
    def buy(self, params: PolicyToAdaptor) -> Optional[float]:
        # Return executed price, or None
        return

    @abstractmethod
    def sell(self, params: PolicyToAdaptor) -> Optional[float]:
        # Return executed price, or None
        return

    @abstractmethod
    def get_price(self) -> float:
        return
    
    @abstractmethod
    def get_leverage(self) -> int:
        return 

    @abstractmethod
    def get_latest_kline_value(self, name: DataElements) -> float:
        return

    @abstractmethod
    def is_next_step(self) -> bool:
        return

    @abstractmethod    
    def balance(self) -> float:
        return
        
    @abstractmethod    
    def pos_amount(self) -> float:
        return

    @abstractmethod
    def entry_price(self) -> float:
        return

    @abstractmethod
    def is_finished(self) -> bool:
        return

    @abstractmethod
    def get_timestamp(self) -> int:
        return

    def get_time_str(self) -> str:
        return milliseconds_to_date(self.get_timestamp())

    def _log(self, s):
        if self.log_en:
            print(s)


class AdaptorBinance(Adaptor):

    class OrderSide(Enum):
        BUY = Client.SIDE_BUY
        SELL = Client.SIDE_SELL
    BUY = OrderSide.BUY
    SELL = OrderSide.SELL

    def __init__(self, usd_name, token_name, data:Data, log_en=True):
        super().__init__(usd_name, token_name, data, log_en)
        proxies = {
            "http": "http://127.0.0.1:8900",
            "https": "http://127.0.0.1:8900",
        }
        self.client = Client(API_KEY, SECRET_KEY, {'proxies': proxies})
        self._update_account_info()
        self.time_minute = self._get_time_minute()
        self.data = data

    def balance(self, refresh=False):
        if refresh:
            self._update_account_info()
        ust_amount = [a for a in self.account_info['assets'] if a['asset'] == self.usd_name][0]
        return float(ust_amount['availableBalance'])

    def pos_amount(self, refresh=False):
        if refresh:
            self._update_account_info()
        pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
        
        return float(pos['positionAmt'])

    def entry_price(self, refresh=False) -> float:
        if refresh:
            self._update_account_info()
        pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
        
        return float(pos['entryPrice'])
    
    def get_leverage(self, refresh=False) -> int:
        if refresh:
            self._update_account_info()
        pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
        return int(pos['leverage'])


    def buy(self, params: PolicyToAdaptor) -> Optional[float]:
        current_price = self.get_price(self.SELL)
        executed_price = None
        if (params.direction == params.ABOVE and current_price > params.price) or (
            params.direction == params.BELLOW and current_price < params.price
        ):
            leverage = self.get_leverage()
            balance = self.balance()
            pos_amount = leverage * balance / current_price
            # pos_amount must be an integer multiple of self.token_min_pos
            pos_amount = pos_amount // self.token_min_pos * self.token_min_pos
            
            assert pos_amount >= self.token_min_pos, 'Pos amount is {}, but min value is {}'.format(
                pos_amount, self.token_min_pos)
            
            if pos_amount >= self.token_min_pos:
                executed_price = self._order_limit_best_price(self.BUY, pos_amount, wait_finished=True)
                self._update_account_info()
            else:
                executed_price = None
        else:
            executed_price = None

        return executed_price

    def sell(self, params: PolicyToAdaptor) -> Optional[float]:
        current_price = self.get_price(self.BUY)
        executed_price = None
        if (params.direction == params.ABOVE and current_price > params.price) or (
            params.direction == params.BELLOW and current_price < params.price
        ):
            pos_amount = self.pos_amount()
            assert pos_amount > 0, 'Pos amount cannot be 0 when sell'
            
            if pos_amount > 0:
                executed_price = self._order_limit_best_price(self.OrderSide.BUY, pos_amount, wait_finished=True)
                self._update_account_info()
            else:
                executed_price = None 
        else:
            executed_price = None

        return executed_price

    def get_price(self, side: OrderSide=OrderSide.SELL) -> float:
        # Price of side sell is higher.
        orderbook_ticket = self._get_orderbook()
        price_type = 'bidPrice' if side == self.OrderSide.BUY else 'askPrice'
        price = float(orderbook_ticket[price_type])
        return price
    
    def get_latest_kline_value(self, name: DataElements) -> float:
        return self.data.get_value(name, -1)

    def is_next_step(self) -> bool:
        time_minute = self._get_time_minute()
        if time_minute > self.time_minute:
            self.time_minute = time_minute
            self.data.update(end_ms=time_minute*60000-1)
            return True
        else:
            return False

    def is_finished(self) -> bool:
        return False
    
    def get_timestamp(self) -> int:
        timestamp = self.client.get_server_time()
        return int(timestamp['serverTime'])

    # ====================================== internal ======================================

    # Change self.account_info
    # Need to call after buying, selling
    def _update_account_info(self):
        # Ref: https://binance-docs.github.io/apidocs/futures/cn/#v2-user_data-2
        self.account_info = self.client.futures_account()

    def _order_limit_best_price(self, side: OrderSide, quantity:float, 
                                wait_finished:Optional[bool]=False) -> float:
        # Do the limit order with dynamic best price. Can choose to wait until all finished 
        # Return (average) price 
        
        left_quantity = quantity
        average_price = 0
        cost = 0

        while True:
            # Choose the best price
            orderbook = self._get_orderbook()
            side_price = float(orderbook['bidPrice']) if side == self.OrderSide.BUY else float(orderbook['askPrice'])
            other_side_price = float(orderbook['askPrice']) if side == self.OrderSide.BUY else float(orderbook['bidPrice'])

            delta = int(side_price * 1e7) - int(other_side_price * 1e7)
            price = (int(side_price * 1e7) + delta) / 1e7

            # Create limit order with best price
            order = self._order_limit(side, left_quantity, price)
            self._log('Best price is {:.4f}'.format(price))
            if self.log_en:
                orderbook['time'] = milliseconds_to_date(orderbook['time'])
                print(orderbook)
            
            if wait_finished:
                # Check order state each 1 second, if can not executed, cancel and create a new one
                
                # 1. Wait until fully executed or price changed
                while True:
                    time.sleep(0.5)
                    executed_amount = self._get_order_executed_amount(order['orderId'])
                    if executed_amount == left_quantity:
                        # Fully executed, then break
                        # self._log('Fully executed, then break')
                        break
                    else:
                        # If not fully executed, check the price
                        if self.get_price(side) != price:
                            # Price changed, no need to wait, break 
                            # self._log('Price changed, no need to wait, break')
                            break

                # 2. Check whether cancel order, and get latest exe amount
                if executed_amount != left_quantity:
                    # Price changed and not fully executed, cancel it and create a new order.
                    # Note that it may be different of the exe amount between when cancel and prior check
                    # self._log('executed_amount is {}, not equal to left_quantity {}, Cancel the order'.format(
                    #     executed_amount, left_quantity))
                    try:
                        executed_amount = self._cancel_order(order['orderId'])
                    except BinanceAPIException:
                        # Order just executed
                        executed_amount = left_quantity

                # 3. Add up the cost
                cost += price * executed_amount

                # 4. Whether fully executed, if not create new order
                if executed_amount == left_quantity:
                    # All quantity are executed, calculate the average price, then break the loop
                    average_price = cost / quantity
                    break
                else:
                    # Not fully executed, calculate the left quantity and continue
                    left_quantity -= executed_amount
                    continue

            else:
                # Don't wait finished, break the loop
                average_price = price
                break
        
        return average_price

    def _order_limit(self, side:OrderSide, quantity:float, price:float):
        return self.client.futures_create_order(
                    symbol = self.symbol, 
                    side = side.value, 
                    type = self.client.FUTURE_ORDER_TYPE_LIMIT, 
                    quantity = quantity, 
                    price = price, 
                    timeInForce = self.client.TIME_IN_FORCE_GTC)

    def _order_market(self, side:OrderSide, quantity:float):
        return self.client.futures_create_order(symbol=self.symbol, side=side.value, 
            type=self.client.FUTURE_ORDER_TYPE_MARKET, quantity=quantity)

    def _get_order_executed_amount(self, order_id) -> float:
        order_info = self.client.futures_get_order(symbol=self.symbol, orderId=order_id)
        return float(order_info['executedQty'])
    
    def _cancel_order(self, order_id) -> float:
        # Return order executed amount
        order_info = self.client.futures_cancel_order(symbol=self.symbol, orderId=order_id)
        return float(order_info['executedQty'])

    def _get_orderbook(self):
        return self.client.futures_orderbook_ticker(symbol=self.symbol)

    def _get_min_leverage(self) -> int:
        price = self.get_price()
        min_btc = 0.001
        min_busd = 0.001 * price
        min_leverage = min_btc * price / self.balance()
        leverage = int(min_leverage) + 1
        self._log('price: {}, min_busd: {:.2f}, min_leverage: {:.2f}, leverage: {}'.format(price, min_busd, min_leverage, leverage))
        return leverage
    
    def _set_leverage(self, leverage: Optional[int]=None):
        if leverage is None:
            leverage = self._get_min_leverage()
        self.client.futures_change_leverage(symbol=self.symbol, leverage=leverage)
        self._update_account_info()

    def _get_time_minute(self) -> int:
        return self.get_timestamp() // 60000


class AdaptorSimulator(Adaptor):

    class PriceType(Enum):
        HIGH = DataElements.HIGH
        LOW = DataElements.LOW
        OPEN = DataElements.OPEN
        CLOSE = DataElements.CLOSE
    HIGH = PriceType.HIGH
    LOW = PriceType.LOW
    OPEN = PriceType.OPEN
    CLOSE = PriceType.CLOSE

    def __init__(self, usd_name: str, token_name: str, init_balance: float, 
                 leverage: int, data: Data, fee: float, log_en=True):
        super().__init__(usd_name, token_name, data, log_en)
        self._balance = init_balance
        self._pos_amount = 0
        self.leverage = leverage
        self.i = 0
        self.fee = fee

        self.price_last_trade = None

    def buy(self, params: PolicyToAdaptor) -> Optional[float]:
        # Return executed price, or None
        price = self._can_buy_or_sell(params)
        if price:
            pos_amount = self.leverage * self._balance / price
            # pos_amount must be an integer multiple of self.token_min_pos
            pos_amount = pos_amount // self.token_min_pos * self.token_min_pos

            assert pos_amount >= self.token_min_pos, 'Pos amount is {}, but min value is {}'.format(
                pos_amount, self.token_min_pos)

            if pos_amount >= self.token_min_pos:
                # Update balance and pos_amount
                self._pos_amount += pos_amount
                total_u = pos_amount * price
                self._balance -= total_u / self.leverage + total_u * self.fee
                self.price_last_trade = price

        return price

    def sell(self, params: PolicyToAdaptor) -> Optional[float]:
        # Return executed price, or None
        price = self._can_buy_or_sell(params)
        if price:
            # Sell all
            assert self._pos_amount > 0, 'Pos amount cannot be 0 when sell'
            assert self.price_last_trade

            bought_balance = self.price_last_trade * self._pos_amount / self.leverage
            earn = (price - self.price_last_trade) * self._pos_amount
            sold_balance = bought_balance + earn
            # sold_balance = self.price_last_trade * self._pos_amount / self.leverage + \
            #                (price - self.price_last_trade) * self.leverage

            self._balance += sold_balance * (1 - self.fee)
            self._pos_amount = 0
            self.price_last_trade = price
        
        return price


    def get_leverage(self) -> int:
        return self.leverage

    def get_price(self, type: PriceType=PriceType.CLOSE) -> float:
        return self.data.get_value(type.value, self.i)
    
    def get_latest_kline_value(self, name: DataElements) -> float:
        return self.data.get_value(name, self.i-1)
    
    def is_next_step(self) -> bool:
        self.i += 1
        return True
    
    def balance(self) -> float:
        return self._balance
        
    def pos_amount(self) -> float:
        return self._pos_amount

    def entry_price(self) -> Optional[float]:
        return self.price_last_trade
    
    def is_finished(self) -> bool:
        return True if self.i >= self.data.len() else False

    def get_timestamp(self) -> int:
        assert self.data.len() > 0, 'Data is empty'
        if self.i >= self.data.len():
            timestamp = self.data.get_value(DataElements.OPEN_TIME, self.i-1) + 60000
        else:
            timestamp = self.data.get_value(DataElements.OPEN_TIME, self.i)    

        return int(timestamp)

    def _can_buy_or_sell(self, params: PolicyToAdaptor) -> Optional[float]:
        # Return executed price, or None
        can_exe = (params.direction == params.BELLOW and self.get_price(self.LOW) < params.price) or (
                   params.direction == params.ABOVE and self.get_price(self.HIGH) > params.price)
        if can_exe:
            opt_fun = min if params.direction == params.BELLOW else max
            price = opt_fun(params.price, self.get_price(self.OPEN))
        else:
            price = None

        return price
