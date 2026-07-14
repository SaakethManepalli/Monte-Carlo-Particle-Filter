from decimal import Decimal, getcontext
import numpy as np
from pylfsr import LFSR
from fxpmath import Fxp

#Variables and definitons 
global seq_full
getcontext().prec = 5
x = Fxp(0, signed=True, n_word=16, n_frac=8)

# Running LFSR
state = [0,0,0,1,0]
fpoly = [5,3]
L = LFSR(fpoly=fpoly, initstate = state)

# Cycling and Converting to fxp
str = ''
for int i in range(10):
    run_full()



print('full period')

print(L.arr2str(seq_full))




x.set_val(str)

x.info(verbose = 3)



'''
LSFR into Binary Array

Binary Array into Binary String

String --> Fixed point

'''

def run_full():
    seq_full = L.runFullPeriod()
    for i in seq_full:
        str += f'{seq_full[i]}'