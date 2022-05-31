from abc import ABC, abstractmethod
from turtle import pos
from typing import Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from enum import Enum
from utils import *

from api_key import API_KEY, SECRET_KEY

class Adaptor(ABC):

    def __init__(self, usd_name, token_name, log_en):
        self.usd_name = usd_name
        self.symbol = token_name + usd_name
        self.log_en = log_en

    @abstractmethod
    def buy(self):
        return

    @abstractmethod
    def sell(self):
        return

    @abstractmethod
    def get_price(self):
        return
    
    @abstractmethod
    def is_next_step(self):
        return
    
    def _log(self, s):
        if self.log_en:
            print(s)


class AdaptorBiance(Adaptor):

    class OrderSide(Enum):
        BUY = Client.SIDE_BUY
        SELL = Client.SIDE_SELL
    BUY = OrderSide.BUY
    SELL = OrderSide.SELL

    def __init__(self, usd_name, token_name, log_en=True):
        proxies = {
            "http": "http://127.0.0.1:8900",
            "https": "http://127.0.0.1:8900",
        }
        self.client = Client(API_KEY, SECRET_KEY, {'proxies': proxies})
        super().__init__(usd_name, token_name, log_en)
        self._update_account_info()

    def balance(self, refresh=False):
        if refresh:
            self._update_account_info()
        ust_amount = [a for a in self.account_info['assets'] if a['asset'] == self.usd_name][0]
        # self._log('ust_amount: {}'.format(ust_amount))
        return float(ust_amount['availableBalance'])

    def pos_amount(self, refresh=False):
        if refresh:
            self._update_account_info()
        pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
        
        return float(pos['positionAmt'])

    def buy(self) -> float:
        # Return bought pos amount
        leverage = self._get_leverage()
        balance = self.balance()
        pos_amount = (leverage * balance * 1000) // 1 / 1000
        min_btc = 0.001
        assert self.symbol == 'BTCBUSD', 'Currently only support BTC'
        assert pos_amount >= min_btc, 'Min pos amount is 0.001'
        
        if pos_amount >= min_btc:
            price = self._order_limit_best_price(self.OrderSide.BUY, pos_amount, wait_finished=True)
            # TODO save price and compare with the expect price.
            self._update_account_info()
        else:
            pos_amount = 0
        
        return pos_amount

    def sell(self):
        # Return sold pos amount
        pos_amount = self.pos_amount()
        assert pos_amount > 0, 'Sell amount cannot be 0'
        
        if pos_amount > 0:
            price = self._order_limit_best_price(self.OrderSide.BUY, pos_amount, wait_finished=True)
            self._update_account_info()

        return pos_amount

    def get_price(self, side: OrderSide=OrderSide.SELL) -> float:
        # Price of side sell is higher.
        orderbook_ticket = self._get_orderbook()
        price_type = 'bidPrice' if side == self.OrderSide.BUY else 'askPrice'
        price = float(orderbook_ticket[price_type])
        return price
    
    def is_next_step(self) -> bool:
        return True
    
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
            price = self.get_price(side)
            # Create limit order with best price
            order = self._order_limit(side, left_quantity, price)
            self._log('Best price is {:.2f}'.format(price))
            if self.log_en:
                orderbook = self._get_orderbook()
                orderbook['time'] = milliseconds_to_date(orderbook['time'])
                print(orderbook)
            
            if wait_finished:
                # Check order state each 1 second, if can not executed, cancel and create a new one
                
                # 1. Wait until fully executed or price changed
                while True:
                    time.sleep(1)
                    executed_amount = self._get_order_executed_amount(order['orderId'])
                    if executed_amount == left_quantity:
                        # Fully executed, then break
                        self._log('Fully executed, then break')
                        break
                    else:
                        # If not fully executed, check the price
                        if self.get_price(side) != price:
                            # Price changed, no need to wait, break 
                            self._log('Price changed, no need to wait, break')
                            break

                # 2. Check whether cancel order, and get latest exe amount
                if executed_amount != left_quantity:
                    # Price changed and not fully executed, cancel it and create a new order.
                    # Note that it may be different of the exe amount between when cancel and prior check
                    self._log('executed_amount is {}, not equal to left_quantity {}, Cancel the order'.format(
                        executed_amount, left_quantity))
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
    
    def _get_leverage(self, refresh=False) -> int:
        if refresh:
            self._update_account_info()
        pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
        return int(pos['leverage'])

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

    