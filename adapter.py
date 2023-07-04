from abc import ABC, abstractmethod
from typing import Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from enum import Enum, auto
from utils import milliseconds_to_date

from api_key import API_KEY, SECRET_KEY
from base_types import OrderSide, DirectionType, Order, DataElements, Recoverable, TradeInfo, DataType
from data import Data

class Adaptor(ABC):

    BUY = OrderSide.BUY
    SELL = OrderSide.SELL

    token_min_pos_table = {
        'BTC': 5,
        'GMT': 2,
        'LUNA2': 0,
        'DOGE': 0,
        '1000LUNC': 0,
        'SOL': 0,
    }

    token_min_price_precision_table = {
        'BTC': 2,
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
        self.min_pos = pow(10, -self.token_min_pos)

        self.log_en = log_en

        self.order_id: int   = 0
        self.order_info      = None

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

    class OrderType(Enum):
        LIMIT     = auto()
        STOP      = auto()
        MARKET    = auto()

    def __init__(self, usd_name, token_name, data:Data, log_en=True, leverage=2, is_futures=True):
        super().__init__(usd_name, token_name, data, log_en)
        proxies = {
            "http": "http://127.0.0.1:8900",
            "https": "http://127.0.0.1:8900",
        }
        self.client = Client(API_KEY, SECRET_KEY, {'proxies': proxies, 'timeout': 20}) # type: ignore
        self.is_futures = is_futures
        self._update_account_info()
        self.enable_oco = True
        
        if self.is_futures:
            if self.get_leverage() <= leverage:
                self._set_leverage(leverage*2)
            self.leverage = leverage
        
        self.leverage = self.get_leverage()

        self.balance(refresh_account=False, update=True)    # Update _balance
        self.time_minute = self._get_time_minute()
        self.data = data
        self.data.set_client(self.client)
        self.price_last_trade = 0

    def reset(self):
        self.clear_open_orders()
        self.price_last_trade = 0
        self.order_id        += 1
        self.order_info       = None

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

    def total_value(self, refresh=False) -> float:
        if refresh:
            self._update_account_info()
        
        if self.is_futures:
            ust_amount = [a for a in self.account_info['assets'] if a['asset'] == self.usd_name][0]
            value = float(ust_amount['marginBalance'])
        else:
            value = self.balance() + self.pos_value()
        
        return value

    def balance(self, refresh_account=False, update=False):
        if refresh_account:
            self._update_account_info()
        
        if self.is_futures:
            if update:
                ust_amount = [a for a in self.account_info['assets'] if a['asset'] == self.usd_name][0]
                self._balance = float(ust_amount['marginBalance']) - self.pos_value()
        else:
            self._balance = float(self.account_info['assets'][0]['quoteAsset']['netAsset'])
        
        return self._balance

    def pos_amount(self, refresh=False):
        if refresh:
            self._update_account_info()
        if self.is_futures:
            pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
            amount = float(pos['positionAmt'])
        else:
            amount = float(self.account_info['assets'][0]['baseAsset']['netAsset'])
        
        return amount

    def pos_value(self, price=None, refresh=False) -> float:
        if price is None:
            price = self.get_price()

        if self.is_futures:
            # Refresh once is enough
            earn = (price - self.entry_price(refresh)) * self.pos_amount()
            value = self.entry_value() + earn
        else:
            value = price * self.pos_amount()

        return value

    def entry_value(self, refresh=False) -> float:
        assert (not self.is_futures), 'entry_value can not used to futures'
        # Position value when opening
        # Refresh once is enough
        pos_amount = self.pos_amount()
        if self.is_futures:
            pos_amount = pos_amount if pos_amount >= 0 else -pos_amount
            value = self.entry_price(refresh) * pos_amount / self.leverage
        else:
            value = self.entry_price(refresh) * pos_amount
        
        return value

    def entry_price(self, refresh=False) -> float:
        if refresh:
            self._update_account_info()
        
        if self.is_futures:
            pos = [a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]
            price = float(pos['entryPrice'])
        else:
            price = self.price_last_trade
        
        return price
    
    def get_leverage(self, refresh=False) -> int:
        if refresh:
            self._update_account_info()
        if self.is_futures:
            leverage = int([a for a in self.account_info['positions'] if a['symbol'] == self.symbol][0]['leverage'])
        else:
            leverage = int(self.account_info['assets'][0]['marginRatio'])
        return leverage

    def create_order(self, order: Order) -> Order.State:
        return self.update_order(order)

    def update_order(self, order: Order) -> Order.State:
        state: Order.State = order.state
        last_state: Order.State = state

        while True:
            last_state = state
            if state == Order.State.CREATED:
                entered_info = order.entered_info
                
                # Whether locked
                timestamp = self.get_timestamp()
                if not entered_info.is_locked(timestamp):
                    self.try_to_trade(entered_info)
                    
                    if entered_info.is_executed():
                        state = Order.State.ENTERED
                    # Not executed, check whether cna be sent.
                    elif entered_info.is_sent():
                        state = Order.State.SENT

            elif state == Order.State.SENT:
                entered_info = order.entered_info
                # It will update self.order_info
                order_info = self._get_order(order_id=entered_info.order_id)

                if self._is_order_filled(order_info):
                    time = self._get_order_time(order_info)
                    assert time, 'time is None'
                    entered_info.executed(self._get_exe_price(order_info), time)
                    state = Order.State.ENTERED

            elif state == Order.State.ENTERED or state == Order.State.EXIT_SENT:
                if order.has_exit():
                    traded      = False
                    sent_info   = None
                    sent        = False

                    # TODO Send two order might cause error
                    # 1. Check sent order first
                    all_exited_infos = order.exited_infos
                    
                    for info in all_exited_infos:
                        if info.is_sent():
                            sent_info = info
                            order_info = self._get_order(order_id=info.order_id)

                            if self._is_order_filled(order_info):
                                time = self._get_order_time(order_info)
                                assert time, 'time is None'
                                info.executed(self._get_exe_price(order_info), time)
                                self._update_account_info()
                                traded = True
                                break
                            
                    conflict_id = sent_info.order_id if sent_info else None

                    # 2. If not traded, check each exit info without send.
                    timestamp = self.get_timestamp()
                    if traded == False:
                        for info in all_exited_infos:
                            if not info.is_locked(timestamp) and self.try_to_trade(info, False, conflict_id):
                                traded = True
                                break
                                

                    # 3. If not traded, check each exit info can be sent, send the info with min price delta.
                    if traded == False:
                        
                        # Find the info closest to the price
                        price = self.get_price()
                        best_price_delta = float('inf')
                        best_info = None
                        
                        for info in all_exited_infos:
                            if not info.is_locked(timestamp) and info.is_sent() == False and info.can_be_sent:
                                delta = abs(price - info.price)
                                if delta < best_price_delta:
                                    best_price_delta = delta
                                    best_info = info

                        # If there is a sent info, calculate whether replace to this one.
                        if sent_info:
                            sent_delta = abs(price - sent_info.price)
                            
                            # If new order is not good enough than sent, not sent this.
                            if best_price_delta > sent_delta * 2 / 3:
                                best_info = None
                                best_price_delta = float('inf')

                        # If need to send a new order info     
                        if best_info:
                            self.try_to_trade(best_info, True, conflict_id)

                            if best_info.is_executed():
                                # Traded successfully
                                traded = True
                            elif best_info.is_sent():
                                sent = True
                    
                            if (traded or sent) and sent_info:
                                sent_info._is_sent = False

                    if traded:
                        state = Order.State.EXITED
                    elif sent:
                        state = Order.State.EXIT_SENT
                    else:
                        # No change, Don't change state
                        pass
                else:
                    # Order don't has exits. Do nothing
                    pass

            else:
                assert (state == Order.State.FINISHED or
                        state == Order.State.CANCELED or
                        state == Order.State.EXITED), 'State error'
                break

            # No state changed, break
            if state == last_state:
                break
            else:
                order.set_state_to(state)

        return state

    def cancel_order(self, order: Order) -> bool:
        canceled = True
        id_need_cancel = None

        state = order.state
        
        # 1. Get the order id need to be canceled
        if state == Order.State.SENT:
            id_need_cancel = order.entered_info.order_id
        
        elif state == Order.State.EXIT_SENT:
            all_exited_infos = order.exited_infos
            
            for info in all_exited_infos:
                if info.is_sent():
                    id_need_cancel = info.order_id
                    break
        
        # 2. If order_id is not none, cancel this order
        if id_need_cancel is not None:
            exe_amount = self._cancel_order(order_id=id_need_cancel)
            
            # Already exe
            if exe_amount > 0:
                canceled = False
            else:
                canceled = True

        if canceled:
            order.set_state_to(Order.State.CANCELED)
        
        return canceled

    def try_to_trade(
            self, 
            trade_info: TradeInfo, 
            can_be_sent: bool=True, 
            conflict_id: Optional[int]=None) -> Optional[float]:
        
        price       = trade_info.price
        direction   = trade_info.direction 
        side        = trade_info.side
        
        # Return executed price, or None
        other_side = OrderSide.BUY if side == OrderSide.SELL else OrderSide.SELL
        current_price = self.get_price(other_side)
        executed_price = None

        # 1. Get the order type
        target_price = price
        if side == OrderSide.BUY:
            if direction == DirectionType.ABOVE and current_price <= price:
                order_type = self.OrderType.STOP
            elif direction == DirectionType.BELLOW and current_price > price:
                order_type = self.OrderType.LIMIT
            else:
                target_price = current_price
                order_type = self.OrderType.MARKET
        else:
            # Sell
            if direction == DirectionType.ABOVE and current_price <= price:
                order_type = self.OrderType.LIMIT
            elif direction == DirectionType.BELLOW and current_price > price:
                order_type = self.OrderType.STOP
            else:
                target_price = current_price
                order_type = self.OrderType.MARKET
        
        # 2. Whether send order to server
        if order_type == self.OrderType.MARKET or (can_be_sent and trade_info.can_be_sent):
            # Can trade immediately or can be sent to server first
            can_create_new = True
            
            # 2.1 Process the conflict order first
            if conflict_id is not None:
                conflict_order_info = self._get_order(order_id=conflict_id)
                
                if not self._is_order_filled(conflict_order_info):
                    # If it is not filled, cancel it
                    exe_amout = self._cancel_order(order_id=conflict_id)
                    assert exe_amout == 0, 'Cancel failed'
                else:
                    # If it is filled, we can not create new.
                    can_create_new = False
                
            # 2.2 Whether can create a new order
            if can_create_new:
                # Reduce amount
                reduce_pos_amount = self.pos_amount()
                reduce_pos_amount = reduce_pos_amount if reduce_pos_amount >= 0 else -reduce_pos_amount
                
                if trade_info.reduce_only == False:
                    leverage = min(self.leverage, trade_info.leverage)
                    pos_amount = leverage * self.total_value() / (target_price * 1.001) - self.min_pos

                else:
                    pos_amount = 0
                
                pos_amount += reduce_pos_amount
                # pos_amount must be an integer multiple of self.token_min_pos
                pos_amount = round(pos_amount, self.token_min_pos)
                
                if pos_amount >= self.min_pos:
                    # Market
                    if order_type == self.OrderType.MARKET:
                        executed_price = self._order_market_wait_finished(side, pos_amount)
                        # executed_price = self._order_limit_best_price(side, pos_amount, wait_finished=True)
                        self.price_last_trade = executed_price
                        self.balance(update=True)

                        exe_time = self._get_order_time(self.order_info)
                        assert exe_time is not None, 'exe_time is None'
                        trade_info.executed(executed_price, exe_time)
                    
                    # Limit
                    elif order_type == self.OrderType.LIMIT:
                        order_info = self._order_limit(side, pos_amount, target_price)
                        assert order_info is not None, 'Order info is None when limit'
                        trade_info.sent(order_info['orderId'])

                    # STOP
                    elif order_type == self.OrderType.STOP:
                        price = target_price * 1.0001 if side == OrderSide.BUY else target_price * 0.999
                        order_info = self._order_stop(side, pos_amount, stopPrice=target_price, price=price)
                        assert order_info is not None, 'Order info is None when stop'
                        trade_info.sent(order_info['orderId'])

                else:
                    executed_price = None
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
        tr = self.data.get_value(DataElements.HIGH, -1) - self.data.get_value(DataElements.LOW, -1)

        while True:
            # Choose the best price
            orderbook = self._get_orderbook()
            low_price = float(orderbook['bidPrice'])
            high_price = float(orderbook['askPrice'])
            mid_price = (low_price + high_price) / 2
            abs_rise = abs(mid_price - last_close)

            if side == OrderSide.BUY:
                price = low_price - max(0.2 * tr, 0.3 * abs_rise)
            else:
                price = high_price + max(0.2 * tr, 0.3 * abs_rise)
            price = round(price, self.token_min_price_precision)

            # Create limit order with best price
            order = self._order_limit(side, left_quantity, price)
            self._log('Best price is {:.4f}'.format(price))
            if self.log_en:
                orderbook['time'] = milliseconds_to_date(order['transactTime'])
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
                        if (side == OrderSide.BUY and self.get_price(side) > high_price) or (
                           (side == OrderSide.SELL and self.get_price(side) < low_price)
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
            print(orderbook)

        while True:
            if self._is_order_filled(order_info):
                break
            else:
                time.sleep(0.5)
                order_info = self._get_order(client_order_id = self.order_id)
                if order_info is None:
                    raise Exception('Order not exist when wait finished after order market')
        
        if self.is_futures:
            price = order_info['avgPrice']
            price = float(price)
        else:
            price = float(order_info['cummulativeQuoteQty']) / float(order_info['executedQty'])
        
        return price

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

    # ------------------------ Utils ------------------------

    def _get_order_time(self, order_info) -> Optional[int]:
        time = None
        if order_info:
            if 'updateTime' in order_info:
                time = order_info['updateTime']
            elif 'transactTime' in order_info:
                time = order_info['transactTime']
            elif 'transactionTime' in order_info:
                time = order_info['transactionTime']
        
        return time
    
    def _is_order_filled(self, order_info) -> bool:
        assert order_info, 'Order info is None when check is order filled'
        return order_info['status'] == 'FILLED'
    
    def _get_exe_price(self, order_info) -> float:
        assert order_info and self._is_order_filled(order_info), 'get exe price failed'

        if self.is_futures:
            price = order_info['avgPrice']
            price = float(price)
        else:
            price = float(order_info['cummulativeQuoteQty']) / float(order_info['executedQty'])

        return price

# ------------------------ Post method ------------------------

    def _order_limit(self, side:OrderSide, quantity:float, price:float):
        return self._order(
            'futures_create_order' if self.is_futures else 'create_margin_order',
            side = side.value, 
            type = 'LIMIT', 
            quantity = quantity, 
            price = round(price, self.token_min_price_precision), 
            timeInForce = self.client.TIME_IN_FORCE_GTC)

    def _order_market(self, side:OrderSide, quantity:float):
        return self._order(
            'futures_create_order' if self.is_futures else 'create_margin_order',
            side = side.value, 
            type = 'MARKET', 
            quantity = quantity)
    
    def _order_stop(self, side:OrderSide, quantity:float, stopPrice:float, price:float):
        o_type = self.client.FUTURE_ORDER_TYPE_STOP if self.is_futures else\
                    self.client.ORDER_TYPE_STOP_LOSS_LIMIT

        return self._order(
            'futures_create_order' if self.is_futures else 'create_margin_order',
            side = side.value, 
            type = o_type, 
            quantity = quantity,
            stopPrice = round(stopPrice, self.token_min_price_precision),
            price = round(price, self.token_min_price_precision),
            timeInForce = self.client.TIME_IN_FORCE_GTC)

    def _order_oco(self, side:OrderSide, quantity:float, price:float, stopPrice:float, stopLimitPrice:float):
        assert not self.is_futures, 'Not support futures'

        return self._order(
            'create_margin_oco_order',
            side = side.value, 
            quantity = quantity,
            price = round(price, self.token_min_price_precision),
            stopPrice = round(stopPrice, self.token_min_price_precision),
            stopLimitPrice = round(stopLimitPrice, self.token_min_price_precision),
            stopLimitTimeInForce = self.client.TIME_IN_FORCE_GTC)

    def _order(self, method, **kwargs):
        while True:
            try:
                kwargs['symbol'] = self.symbol
                
                self.order_id += 1
                if 'oco' in method:
                    kwargs['listClientOrderId'] = self.order_id    
                else:
                    kwargs['newClientOrderId'] = self.order_id

                if not self.is_futures:
                    kwargs['isIsolated'] = 'TRUE'
                    kwargs['sideEffectType'] = 'MARGIN_BUY' if kwargs['side'] == 'BUY' else 'AUTO_REPAY'

                order_info = getattr(self.client, method)(**kwargs)
                self.order_info = order_info
                self._update_account_info()
                
                price = kwargs['stopPrice'] if 'stopPrice' in kwargs else kwargs['price'] if 'price' in kwargs else 'None'
                self._log(f"{self.get_time_str()}: --- Send order: {kwargs['side']} {kwargs['type']}, price: {price}")

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

    def _cancel_order(self, order_id = None, client_order_id = None) -> float:
        assert order_id or client_order_id, 'No order id'

        # Return order executed amount
        order_info = self._client_call(
            'futures_cancel_order', 
            symbol            = self.symbol, 
            orderId           = order_id, 
            origClientOrderId = client_order_id)
        
        # Update order info
        if client_order_id == self.order_id:
            self.order_info = order_info

        self._update_account_info()

        return float(order_info['executedQty'])

    def _set_leverage(self, leverage: Optional[int]=None):
        assert self.is_futures, 'Only support futures'

        if leverage is None:
            leverage = self._get_min_leverage()
        
        self._client_call('futures_change_leverage', symbol=self.symbol, leverage=leverage)
        self._update_account_info()

    # ------------------------ Get method ------------------------
    
    def _get_order(self, order_id = None, client_order_id = None, is_oco = False):
        assert order_id or client_order_id, 'Order id is None'
        assert not self.is_futures or is_oco == False, 'oco not support for futures'
        # TODO OCO support
        try:
            order_info = self._client_call(
                'futures_get_order', 
                symbol = self.symbol, 
                orderId = order_id, 
                origClientOrderId = client_order_id)
            
            # Update order info
            if client_order_id == self.order_id:
                self.order_info = order_info

            self._update_account_info()
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
        
        def add_p(k, names, values):
            for i in range(len(names)):
                k[names[i]] = values[i]
            return k
        
        get_margin_params = {
            'futures_account': lambda k: ['get_isolated_margin_account', add_p(k, ['symbols'], [self.symbol])],
            'futures_get_open_orders': lambda k: ['get_open_margin_orders', add_p(k, ['isIsolated'], ['TRUE'])],
            'get_server_time': lambda k: ['get_server_time', k],
            'futures_cancel_order': lambda k: ['cancel_margin_order', add_p(k, ['isIsolated'], ['TRUE'])],
            'futures_change_leverage': lambda k: ['', k],
            'futures_get_order': lambda k: ['get_margin_order', add_p(k, ['isIsolated'], ['TRUE'])],
            'futures_orderbook_ticker':lambda k: ['get_orderbook_ticker', k]}
        
        if not self.is_futures:
            assert method in get_margin_params, 'method not exist'
            method, kwargs = get_margin_params[method](kwargs)


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

            if state == Order.State.CREATED or state == Order.State.SENT:
                if not entered_info.is_locked(self.get_timestamp()):
                    # Can not be sent, Only move to entered when trade successfully
                    if (self.try_to_trade(entered_info) is not None):
                        state = Order.State.ENTERED

            elif state == Order.State.ENTERED or state == Order.State.EXIT_SENT:
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
                        state == Order.State.EXITED), 'State not correct'
                break

            # No state changed, break
            if state == last_state:
                break
            else:
                order.set_state_to(state)

        return state

    def cancel_order(self, order: Order) -> bool:
        assert order.not_entered(), 'Can not cancel due to order is entered'
        order.set_state_to(Order.State.CANCELED)
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


if __name__ == "__main__":
    usd_name = 'TUSD'
    token_name = 'BTC'
    is_futures = False
    symbol = token_name+usd_name
    data = Data(symbol, DataType.INTERVAL_1MINUTE, num=100, is_futures=is_futures)
    adaptor = AdaptorBinance(usd_name=usd_name, token_name=token_name, data=data, log_en=True, is_futures=is_futures)
    data.update(end_str="1 minute ago UTC+8")
    open_time = adaptor.get_timestamp()
    price = adaptor.get_price(adaptor.BUY)
    order = Order(OrderSide.BUY, 0, Order.ABOVE, 'Long', 
                        open_time, leverage=5, can_be_sent=True, reduce_only=False)

    order.add_exit(price+10, Order.ABOVE, "Long exit")
    order.add_exit(price-100, Order.BELLOW, "Long stop", can_be_sent=True)

    print(order.state)
    adaptor.update_order(order=order)
    # adaptor.cancel_order(order=order)
    print(order.state)