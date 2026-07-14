from decimal import Decimal, getcontext
import numpy as np
from pylfsr import LFSR
getcontext().prec = 5

state = [0,0,0,1,0]
fpoly = [5,3]
L = LFSR(fpoly=fpoly, initstate = state)


seq_full = L.runFullPeriod()

print('full period')

print(L.arr2str(seq_full))


