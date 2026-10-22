"""Thread-safe byte-bounded immutable cache for CPU pipeline intermediates."""
from collections import OrderedDict
import threading
import numpy as np

def size_bytes(value):
    if isinstance(value,np.ndarray): return value.nbytes
    if isinstance(value,dict): return sum(size_bytes(v) for v in value.values())
    if isinstance(value,(list,tuple)): return sum(size_bytes(v) for v in value)
    return 128

def freeze(value):
    if isinstance(value,np.ndarray): value.flags.writeable=False
    elif isinstance(value,dict):
        for v in value.values(): freeze(v)
    elif isinstance(value,(list,tuple)):
        for v in value: freeze(v)

class ByteLRU:
    def __init__(self,budget):
        self.budget=budget; self.used=0; self.items=OrderedDict(); self.lock=threading.RLock(); self.hits=0; self.misses=0
    def compute(self,key,factory):
        with self.lock:
            if key in self.items:
                self.hits+=1;value,n=self.items.pop(key);self.items[key]=(value,n);return value
            self.misses+=1
            value=factory();n=size_bytes(value)
            if n<=self.budget:
                freeze(value)
                while self.used+n>self.budget and self.items:
                    _,(_,old)=self.items.popitem(last=False);self.used-=old
                self.items[key]=(value,n);self.used+=n
            return value
    def clear(self):
        with self.lock: self.items.clear();self.used=0;self.hits=self.misses=0
    def stats(self):
        with self.lock: return {'bytes':self.used,'budget':self.budget,'entries':len(self.items),'hits':self.hits,'misses':self.misses}
