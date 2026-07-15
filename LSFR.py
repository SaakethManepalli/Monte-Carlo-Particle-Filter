from decimal import Decimal, getcontext
import numpy as np
from pylfsr import LFSR
from fxpmath import Fxp


# GLOBALS

N_WORD = 16
N_FRAC = 5


state = [0,0,0,1,0,0,0,1]
fpoly = [8,6,5,4]
L = LFSR(fpoly=fpoly, initstate=state)

# DEFINITIONS
def run_LFSR(machine, n_bits = 8):
    val = 0
    for i in range(8):
        machine.next()
        val = (val << 1) | int(machine.outbit)
    return val
def Fxp_convert(value):
    return Fxp(value, signed=True, n_word=N_WORD, n_frac=N_FRAC)



NUM_SAMPLES = 10
raw_samples = []
fxp_samples = []

for _ in range(NUM_SAMPLES):
    sample = run_LFSR(L, n_bits=8)      # raw integer 0..255
    raw_samples.append(sample)
    fxp_samples.append(Fxp_convert(sample))


print("Fxp samples:", [f() for f in fxp_samples])