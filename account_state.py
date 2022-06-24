from sqlite3 import adapt
from adapter import Adaptor
from base_types import DataElements, IdxValue
from data import Data
import os
import json

class AccountState:

    def __init__(self, adaptor: Adaptor, analyze_en: bool=True, log_en=True) -> None:
        self.balance: float = adaptor.balance()
        self.init_balance: float = self.balance
        self.pos_amount: float = adaptor.pos_amount()
        self.pos_value: float = adaptor.pos_value()
        self.adaptor: Adaptor = adaptor
        self.analyze_en: bool = analyze_en
        self.log_en: bool = log_en
        if self.log_en:
            print('Balance: {:.4f}, pos value, amount: {}, {}'.format(self.balance, self.pos_value, self.pos_amount))

        if self.analyze_en:
            self.update_times = [adaptor.get_timestamp()]
            self.balance_history = [self.balance]
            self.pos_amount_history = [self.pos_amount]
            self.pos_value_history = [self.pos_value]
            self.pos_entry_prices = [adaptor.entry_price()]
            self.pos_entry_values = [adaptor.entry_value()]
            self.earn_points = IdxValue()
    
    def can_buy(self) -> bool:
        if self.pos_amount <= 0:
            return True
        else:
            return False
    
    def can_sell(self) -> bool:
        if self.pos_amount >= 0:
            return True
        else:
            return False
    
    def update_analyzed_info(self, timestamp: int) -> None:
        # Save analyze info
        if self.analyze_en:
            self.update_times.append(timestamp)
            self.balance_history.append(self.balance)
            self.pos_amount_history.append(self.pos_amount)
            self.pos_value_history.append(self.pos_value)
            self.pos_entry_prices.append(self.adaptor.entry_price())
            self.pos_entry_values.append(self.adaptor.entry_value())

    def update(self) -> None:
        self.balance: float = self.adaptor.balance()
        self.pos_amount: float = self.adaptor.pos_amount()
        self.pos_value: float = self.adaptor.pos_value()
        if self.log_en:
            print('Balance: {:.4f}, pos value, amount: {}, {}'.format(self.balance, self.pos_value, self.pos_amount))
            print()
    
    def earn_rate(self, price: float) -> float:
        # 1 represent 100% means no earn
        return self.earn(price) / self.init_balance

    def earn(self, price: float) -> float:
        self.pos_value = self.adaptor.pos_value(price)
        balance = self.balance + self.pos_value
        total_earn = balance - self.init_balance
        return total_earn

    def save_earn_points(self, data: Data, file_loc: str, symbol: str, start, end):
        if self.analyze_en:
            self.get_earn_points(data)
            
            vertices = {
                'earn_idx': self.earn_points.idx,
                'earn_value': self.earn_points.value,
            }

            file_path = os.path.join(file_loc, '{}_start_{}_end_{}_earn_points.json'.format(symbol, start, end))
            with open(file_path, 'w') as f:
                json.dump(vertices, f, indent=2)

    def get_earn_points(self, data: Data) -> IdxValue:
        earn_points = IdxValue()
        if self.analyze_en:
            if self.earn_points.is_empty():
                assert len(self.update_times) == len(self.balance_history) == \
                    len(self.pos_amount_history) == len(self.pos_entry_values) == \
                    len(self.pos_entry_prices)

                # Get init eran_points' idx and the value before first trade
                earn_points.idx = data.data.index.to_list()
                earn_points.value = [0.0] * data.len()
                init_balance = self.balance_history[0]
                i = 0
                for j in range(len(self.update_times)):

                    # 1. Calculate earn of current timestamp
                    if j > 0:
                        earn_points.value[i] = (self.pos_value_history[j] + self.balance_history[j]) / init_balance
                    else:
                        earn_points.value[i] = 1
                    i += 1

                    # 2. Calculate earn until next update
                    close = data.get_value(DataElements.CLOSE, i-1)
                    while (j+1 < len(self.update_times) and \
                            self.update_times[j+1] > data.get_value(DataElements.OPEN_TIME, i)) or \
                        (j+1 >= len(self.update_times) and i < data.len()):

                        if self.pos_amount_history[j] != 0:
                            last_close = close
                            close = data.get_value(DataElements.CLOSE, i)
                            earn = (close - last_close) * self.pos_amount_history[j]
                            earn_rate = earn / init_balance
                        else:
                            earn_rate = 0
                        earn_points.value[i] = earn_points.value[i-1] + earn_rate
                        i += 1

                # loop for each states

                self.earn_points = earn_points
            # if not self.earn_points:
            else:
                # Don't generate earn points again
                earn_points = self.earn_points
                pass

        # if self.analyze_en:
 
        return earn_points

