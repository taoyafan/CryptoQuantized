from abc import ABC, abstractmethod
from typing import Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from enum import Enum
from utils import milliseconds_to_date

from api_key import API_KEY, SECRET_KEY
from base_types import OrderSide, DirectionType, Order, DataElements, Recoverable, TradeInfo
from data import Data

class Adaptor(ABC):

    BUY = OrderSide.BUY
    SELL = OrderSide.SELL

    token_min_pos_table = {
        'BTC': 0.001,
        'GMT': 0.1,
        'LUNA2': 1,
        'DOGE': 1,
        '1000LUNC': 1,
        'SOL': 1,
    }

    token_min_price_precision_table = {
        'BTC': 1,
        'GMT': 4,
        'LUNA2': 4,
        'DOGE': 5,
        '1000LUNC': 4,
        'SOL': 3,
    }

    def __init__(self, usd_name, token_name, data:Data, log_en):
        self.usd_name = usd_name
        self.token_name = token_name
        self.symbol = token_name + usd_name
        self.data = data

        assert token_name in self.token_min_pos_table, 'Not support token'
        self.token_min_pos = self.token_min_pos_table[token_name]
        self.token_min_price_precision = self.token_min_price_precision_table[token_name]
        self.log_en = log_en

        self.order_id = 0

    @abstractmethod
    def create_order(self, order: Order) -> Order.State:
        return

    @abstractmethod
    def update_order(self, order: Order) -> Order.State:
        return

    @abstractmethod
    def cancel_order(self, order: Order) -> bool:
        return

    @abstractmethod
    def try_to_trade(self, price: float, direction: DirectionType, 
                     side: OrderSide, reduce_only: bool) -> Optional[float]:
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
    def pos_value(self, price=None) -> float:
        return

    @abstractmethod
    def entry_value(self) -> float:
        # Position value when opening
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

    def __init__(self, usd_name, token_name, data:Data, log_en=True, leverage=2):
        super().__init__(usd_name, token_name, data, log_en)
        proxies = {
            "http": "http://127.0.0.1:8900",
            "https": "http://127.0.0.1:8900",
        }
        self.client = Client(API_KEY, SECRET_KEY, {'proxies': proxies, 'timeout': 20})

        self._update_account_info()
        
        if self.get_leverage() <= leverage:
            self._set_leverage(leverage*2)
        self.leverage = leverage

        self.balance(refresh_account=False, update=True)    # Update _balance
        self.time_minute = self._get_time_minute()
        self.data = data


    def clear_open_orders(self):
        open_orders = self._client_call('futures_get_open_orders', symbol=self.symbol)
        for order in open_orders:
            try:
                self._cancel_order(order_id = order['orderId'])
            except KeyboardInterrupt as ex:
                raise ex
            except:
                # Assume the order is executed
                pass
        self._update_account_info()

    def total_value(self, refresh=False) -> float:
        if refresh:
            self._update_account_info()
        ust_amount = [a for a in self.account_info['assets'] if a['asset'] == self.usd_name][0]
        return float(ust_amount['marginBalance'])

    def balance(self, refresh_account=False, update=False):
        if refresh_account:
            self._update_account_info()
        
        if update:
            ust_amount = [a for a in self.account_info['assets'] if a['asset'] == self.usd_name][0]
            self._balance = float(ust_amount['marginBalance']) - self.pos_value()
        
        return self._balance

    def pos_amount(self, refresh=False):
        if refresh:
            self._update_account_info()
        pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
        
        return float(pos['positionAmt'])

    def pos_value(self, price=None, refresh=False) -> float:
        if price is None:
            price = self.get_price()

        # Refresh once is enough
        earn = (price - self.entry_price(refresh)) * self.pos_amount()
        value = self.entry_value() + earn
        return value

    def entry_value(self, refresh=False) -> float:
        # Position value when opening
        # Refresh once is enough
        pos_amount = self.pos_amount()
        pos_amount = pos_amount if pos_amount >= 0 else -pos_amount
        return self.entry_price(refresh) * pos_amount / self.leverage

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

    # TODO
    def create_order(self, order: Order) -> Order.State:
        raise NotImplementedError(self.__class__.__name__) 

    # TODO
    def update_order(self, order: Order) -> Order.State:
        raise NotImplementedError(self.__class__.__name__) 

    # TODO
    def cancel_order(self, order: Order) -> bool:
        raise NotImplementedError(self.__class__.__name__) 

    def try_to_trade(self, price: float, direction: DirectionType, 
                    side: OrderSide, reduce_only: bool) -> Optional[float]:
        # Return executed price, or None
        other_side = OrderSide.BUY if side == OrderSide.SELL else OrderSide.SELL
        current_price = self.get_price(other_side)
        executed_price = None
        if (direction == DirectionType.ABOVE and current_price > price) or (
            direction == DirectionType.BELLOW and current_price < price
        ):
            # Reduce amount
            reduce_pos_amount = self.pos_amount()
            reduce_pos_amount = reduce_pos_amount if reduce_pos_amount >= 0 else -reduce_pos_amount
            
            if reduce_only == False:
                leverage = self.leverage
                balance = self.balance() + self.pos_value()
                pos_amount = leverage * balance / current_price
                # pos_amount must be an integer multiple of self.token_min_pos
                pos_amount = pos_amount // self.token_min_pos * self.token_min_pos

            else:
                pos_amount = 0
            
            pos_amount += reduce_pos_amount
            
            if pos_amount >= self.token_min_pos:
                # executed_price = self._order_market_wait_finished(side, pos_amount)
                executed_price = self._order_limit_best_price(side, pos_amount, wait_finished=True)
                self._update_account_info()
                self.balance(update=True)
            else:
                executed_price = None
        else:
            executed_price = None

        return executed_price

    def get_price(self, side: OrderSide=OrderSide.SELL) -> float:
        # Price of side sell is higher.
        orderbook_ticket = self._get_orderbook()
        price_type = 'bidPrice' if side == OrderSide.BUY else 'askPrice'
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
        timestamp = self._client_call('get_server_time')
        return int(timestamp['serverTime'])

    # ====================================== internal ======================================

    # ------------------------ Change internal state ------------------------

    # Change self.account_info
    # Need to call after buying, selling
    def _update_account_info(self):
        # Ref: https://binance-docs.github.io/apidocs/futures/cn/#v2-user_data-2
        self.account_info = self._client_call('futures_account')

    # ------------------------ High level ------------------------

    def _order_limit_best_price(self, side: OrderSide, quantity:float, 
                                wait_finished:Optional[bool]=False) -> float:
        # Do the limit order with dynamic best price. Can choose to wait until all finished 
        # Return (average) price 
        
        left_quantity = quantity
        average_price = 0
        cost = 0
        last_close = self.data.get_value(DataElements.CLOSE, -1)

        while True:
            # Choose the best price
            orderbook = self._get_orderbook()
            side_price = float(orderbook['bidPrice']) if side == OrderSide.BUY else float(orderbook['askPrice'])
            other_side_price = float(orderbook['askPrice']) if side == OrderSide.BUY else float(orderbook['bidPrice'])

            # delta = last_close - other_side_price
            # price = side_price + (2 * delta)
            price = (side_price - last_close) * 0.7 + last_close
            price = round(price, self.token_min_price_precision)

            # Create limit order with best price
            order = self._order_limit(side, left_quantity, price)
            self._log('Best price is {:.4f}'.format(price))
            if self.log_en:
                orderbook['time'] = milliseconds_to_date(orderbook['time'])
                orderbook['last_close'] = last_close
                print(orderbook)
            
            if wait_finished:
                # Check order state each 1 second, if can not executed, cancel and create a new one
                
                # 1. Wait until fully executed or price changed
                while True:
                    time.sleep(5)
                    executed_amount = self._get_order_executed_amount(client_order_id = self.order_id)
                    if executed_amount == left_quantity:
                        # Fully executed, then break
                        # self._log('Fully executed, then break')
                        break
                    else:
                        # If not fully executed, check the price
                        if (side == OrderSide.BUY and self.get_price(side) > other_side_price) or (
                           (side == OrderSide.SELL and self.get_price(side) < other_side_price)
                        ):
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

    def _order_market_wait_finished(self, side:OrderSide, quantity:float) -> float:
        order_info = self._order_market(side, quantity)
        if self.log_en:
            orderbook = self._get_orderbook()
            orderbook['time'] = milliseconds_to_date(orderbook['time'])
            print(orderbook)

        while True:
            if order_info['status'] == 'FILLED':
                break
            else:
                time.sleep(0.5)
                order_info = self._get_order(client_order_id = self.order_id)
                if order_info is None:
                    raise Exception('Order not exist when wait finished after order market')

        return float(order_info['avgPrice'])

    def _get_min_leverage(self) -> int:
        price = self.get_price()
        min_busd = self.token_min_pos * price
        min_leverage = self.token_min_pos * price / self.balance()
        leverage = int(min_leverage) + 1
        self._log('price: {}, min_busd: {:.2f}, min_leverage: {:.2f}, leverage: {}'.format(
            price, min_busd, min_leverage, leverage))
        return leverage
    
    def _get_time_minute(self) -> int:
        return self.get_timestamp() // 60000
        
# ------------------------ Post method ------------------------

    def _order_limit(self, side:OrderSide, quantity:float, price:float):
        return self._order(
            symbol = self.symbol, 
            side = side.value, 
            type = self.client.FUTURE_ORDER_TYPE_LIMIT, 
            quantity = quantity, 
            price = round(price, self.token_min_price_precision), 
            timeInForce = self.client.TIME_IN_FORCE_GTC)

    def _order_market(self, side:OrderSide, quantity:float):
        return self._order(
            symbol = self.symbol, 
            side = side.value, 
            type = self.client.FUTURE_ORDER_TYPE_MARKET, 
            quantity = quantity)

    def _order(self, **kwargs):
        while True:
            try:
                self.order_id += 1
                kwargs['newClientOrderId'] = self.order_id
                order_info = self.client.futures_create_order(**kwargs)
                break
            except KeyboardInterrupt as ex:
                raise ex
            except BinanceAPIException as ex:
                print(str(ex) + ', when calling {}, params: {}'.format('_order_market', kwargs))
                # Don't handle here
                raise ex
            except Exception as ex:
                print(str(ex) + ', when calling {}, params: {}'.format('_order_market', kwargs))
                # Error due to network. Check order whether exist, if exist, break
                order_info = self._get_order(client_order_id = self.order_id)
                if order_info:
                    # Order submit succeed
                    break
                else:
                    # Order submit failed, try again
                    time.sleep(1)
                
        return order_info

    def _cancel_order(self, order_id) -> float:
        # Return order executed amount
        order_info = self._client_call('futures_cancel_order', symbol=self.symbol, orderId=order_id)
        return float(order_info['executedQty'])

    def _set_leverage(self, leverage: Optional[int]=None):
        if leverage is None:
            leverage = self._get_min_leverage()
        self._client_call('futures_change_leverage', symbol=self.symbol, leverage=leverage)
        self._update_account_info()

    # ------------------------ Get method ------------------------
    
    def _get_order(self, order_id = None, client_order_id = None):
        assert order_id or client_order_id
        try:
            order_info = self._client_call(
                'futures_get_order', 
                symbol = self.symbol, 
                orderId = order_id, 
                origClientOrderId = client_order_id)
        except KeyboardInterrupt as ex:
            raise ex
        except BinanceAPIException:
            order_info = None

        return order_info

    def _get_order_executed_amount(self, client_order_id) -> float:
        order_info = self._get_order(client_order_id = client_order_id)
        if order_info is None:
            raise Exception('Order not exist when get executed amount')
        return float(order_info['executedQty'])

    def _get_orderbook(self):
        return self._client_call('futures_orderbook_ticker', symbol=self.symbol)

    # ------------------------ Client wrapper ------------------------

    def _client_call(self, method, **kwargs):
        call_cnt = 0
        while True:
            try:
                result = getattr(self.client, method)(**kwargs)
                break
            except KeyboardInterrupt as ex:
                raise ex
            except Exception as ex:
                # Due to network, Try again later
                print(str(ex) + ', when calling {}, params: {}'.format(method, kwargs))
                call_cnt += 1
                # If stilled failed after 3 times calling, raise again
                if call_cnt > 3:
                    print('Still failed after 3 times calling, stop calling')
                    raise(ex)
                else:
                    time.sleep(1)
        
        return result


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

        self.price_last_trade = 0

    def create_order(self, order: Order) -> Order.State:
        return self.update_order(order)

    def update_order(self, order: Order) -> Order.State:
        state: Order.State = order.state
        last_state: Order.State = state
        entered_info = order.entered_info

        while True:
            last_state = state

            if state == Order.State.CREATED or state == Order.State.SENDED:
                if not entered_info.is_locked(self.get_timestamp()):
                    # Can not send, Only move to entered when trade successfully
                    if (self.try_to_trade(entered_info) is not None):
                        state = Order.State.ENTERED

            elif state == Order.State.ENTERED or state == Order.State.EXIT_SENDED:
                if order.has_exit():
                    all_exited_infos = order.exited_infos
                    traded = False

                    for info in all_exited_infos:
                        if not info.is_locked(self.get_timestamp()) and self.try_to_trade(info):
                            # Traded successfully
                            traded = True
                            break

                    if traded:
                        state = Order.State.EXITED

            else:
                assert (state == Order.State.FINISHED or
                        state == Order.State.CANCELED or
                        state == Order.State.EXITED)
                break

            # No state changed, break
            if state == last_state:
                break
            else:
                order.set_state_to(state)

        return state

    def cancel_order(self, order: Order) -> bool:
        assert order.not_entered()
        return True

    def try_to_trade(self, trade_info: TradeInfo) -> Optional[float]:
        price       = trade_info.price
        direction   = trade_info.direction
        side        = trade_info.side
        reduce_only = trade_info.reduce_only
        leverage    = trade_info.leverage

        # Return executed price, or None
        trade_price = self._can_buy_or_sell(price, direction)
        if trade_price:
            trade_info.executed(trade_price, self.get_timestamp())
            # Reduce first
            if self.pos_amount() != 0:
                # pos_value must called before clear pos_amount
                self._balance += self.pos_value(price=trade_price) * (1 - self.fee)
                self._pos_amount = 0
                self.price_last_trade = trade_price

            if reduce_only == False:
                balance = self._balance * 0.95
                self.leverage = leverage
                pos_amount = self.leverage * balance / trade_price
                # pos_amount must be an integer multiple of self.token_min_pos
                pos_amount = pos_amount // self.token_min_pos * self.token_min_pos

                assert pos_amount >= self.token_min_pos, 'Pos amount is {}, but min value is {}'.format(
                pos_amount, self.token_min_pos)

                if pos_amount >= self.token_min_pos:
                    # Update balance and pos_amount
                    self._pos_amount += pos_amount if side == self.BUY else -pos_amount
                    total_u = pos_amount * trade_price
                    self._balance -= total_u / self.leverage + total_u * self.fee
                    self.price_last_trade = trade_price

        return trade_price

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

    def pos_value(self, price=None) -> float:
        if price is None:
            price = self.get_price()
            
        if self.price_last_trade > 0:
            earn = (price - self.price_last_trade) * self.pos_amount()
            value = self.entry_value() + earn
        else:
            value = 0
            
        return value

    def entry_value(self) -> float:
        pos_amount = self._pos_amount if self._pos_amount >= 0 else -self._pos_amount
        return self.price_last_trade * pos_amount / self.leverage

    def entry_price(self) -> float:
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

    def _can_buy_or_sell(self, price: float, direction: DirectionType) -> Optional[float]:
        # Return executed price, or None
        can_exe = (direction == DirectionType.BELLOW and self.get_price(self.LOW) <= price) or (
                   direction == DirectionType.ABOVE and self.get_price(self.HIGH) >= price)
        if can_exe:
            opt_fun = min if direction == DirectionType.BELLOW else max
            trade_price = opt_fun(price, self.get_price(self.OPEN))
        else:
            trade_price = None

        return trade_price
