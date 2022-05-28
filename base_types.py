from typing import Dict, Optional, List

class IdxValue:
    def __init__(self, idx: Optional[List]=None, value: Optional[List]=None):
        self.idx = idx if idx else []
        self.value = value if value else []
    
    def add(self, idx, value):
        self.idx.append(idx)
        self.value.append(value)

class OptPoints:
    def __init__(self):
        self.buy = IdxValue()
        self.sell = IdxValue()

