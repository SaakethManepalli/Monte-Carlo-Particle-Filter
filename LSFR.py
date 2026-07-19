from decimal import Decimal, getcontext
import numpy as np
from pylfsr import LFSR
from fxpmath import Fxp


# GLOBALS
#

N_WORD = 16
N_FRAC = 5
SCALE  = 1 << N_FRAC              # 32 for Q10.5
MAX_VAL = (1 << (N_WORD - 1)) - 1 #  32767
MIN_VAL = -(1 << (N_WORD - 1))    # -32768
M = 32


state = [0,0,0,1,0,0,0,1]
fpoly = [8,6,5,4]
L = LFSR(fpoly=fpoly, initstate=state)

# FIXED POINT CONVERSION 
def sat(x):
    """Saturating clamp to 16-bit signed range — mirrors hardware overflow."""
    return max(MIN_VAL, min(MAX_VAL, x))

def to_q105(x):
    """Float → Q10.5 raw integer. Round + saturate, exactly as a register would."""
    return sat(int(x * SCALE)) #usese int so that it truncates, not banker's round

def from_q105(fx):
    """Q10.5 raw integer → float. For printing and analysis only."""
    return fx / SCALE

def fx_add(a, b):
    """Integer add with saturation (operates on raw Q10.5 ints)."""
    return sat(a + b)

def fx_sub(a, b):
    """Integer subtract with saturation (operates on raw Q10.5 ints)."""
    return sat(a - b)

def propogate(particles, machine):
    return [fx_add(p, noise_sample(machine)) for p in particles]

# DEFINITIONS
def run_LFSR(machine, n_bits = 8):
    val = 0
    for i in range(8):
        machine.next()
        val = (val << 1) | int(machine.outbit)
    return val
def Fxp_convert(value):
    return Fxp(value, signed=True, n_word=N_WORD, n_frac=N_FRAC)

def noise_sample(machine):
    total       = sum(run_LFSR(machine) for _ in range(4))  
    noise_float = (total - 510) / 256.0                      # center + scale, ~±2
    return to_q105(noise_float)                              # → Q10.5 raw int

DELTA = to_q105(1.5)
def weight(particles, z):
    """
    Rectangular likelihood: weight = 1 if |z - particle| <= DELTA, else 0.
    z is the measurement in Q10.5. All arithmetic on raw integers.
    Mirrors the Verilog weight module: subtractor + abs + comparator.
    """
    weights = []
    for p in particles:
        diff = abs(fx_sub(z, p))
        weights.append(1 if diff <= DELTA else 0)
    return weights



NUM_SAMPLES = 10
raw_samples = []
fxp_samples = []

for _ in range(NUM_SAMPLES):
    sample = run_LFSR(L, n_bits=8)      # raw integer 0..255
    raw_samples.append(sample) 
    fxp_samples.append(Fxp_convert(sample))


#print("Fxp samples:", [f() for f in fxp_samples])

raw_samples = [run_LFSR(L) for _ in range(10)]
print("Raw LFSR bytes:", raw_samples)

# 2. noise check — what actually feeds the filter
noise = [noise_sample(L) for _ in range(1000)]
mean  = sum(noise) / len(noise)
print(f"\nNoise samples (raw Q10.5 int):")
print(f"  mean: {mean:.2f}   (want ≈ 0)")
print(f"  min:  {min(noise)}   max: {max(noise)}")

# 3. show a few as human-readable floats
print(f"\nNoise as Q10.5 floats:")
print(f"  {[from_q105(n) for n in noise]}") #n is our noise and what we add to particles
print(f"\n Noise + Sample:")
print(f" {[from_q105(n) + sample for n in noise]})")

# ─── WEIGHT TEST ──────────────────────────────────────────────────────────────
print("\n─── Weight test ───")
test_particles = [to_q105(x) for x in [0.0, 1.0, 1.4, 1.6, 3.0, -1.0]]
test_z = to_q105(1.5)     # measurement at 1.5

w = weight(test_particles, test_z)

print(f"DELTA = {from_q105(DELTA)}  (window is ±{from_q105(DELTA)})")
print("particle | distance from z | weight")
for p, wt in zip(test_particles, w):
    dist = abs(from_q105(test_z) - from_q105(p))
    print(f"  {from_q105(p):+5.2f}   |     {dist:.2f}       |   {wt}")