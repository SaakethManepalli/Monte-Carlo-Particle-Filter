import numpy as np

# Fxp
FRAC_BITS   = 8          # fractional bits
SCALE       = 1 << FRAC_BITS          # 256
INT_BITS    = 7          # integer bits (excl. sign)
TOTAL_BITS  = 16         # signed Q8.8
MAX_VAL     = (1 << (TOTAL_BITS - 1)) - 1   #  32767
MIN_VAL     = -(1 << (TOTAL_BITS - 1))       # -32768

def to_fixed(x):
    """Convert float to Q8.8 integer representation."""
    return int(np.clip(np.round(x * SCALE), MIN_VAL, MAX_VAL))

def to_float(fx):
    """Convert Q8.8 integer back to float for analysis."""
    return fx / SCALE

def fixed_add(a, b):
    """Saturating fixed-point add — mirrors hardware behavior."""
    result = a + b
    return int(np.clip(result, MIN_VAL, MAX_VAL))

def fixed_mul(a, b):
    """Q8.8 × Q8.8 → Q8.8: multiply then right-shift by FRAC_BITS."""
    result = (a * b) >> FRAC_BITS
    return int(np.clip(result, MIN_VAL, MAX_VAL))

# ─── LFSR (bit-accurate 8-bit maximal length) ──────────────────────────
def lfsr8_next(state):
    """One step of 8-bit LFSR, taps at 8,6,5,4. Returns next state."""
    feedback = ((state >> 7) ^ (state >> 5) ^
                (state >> 4) ^ (state >> 3)) & 1
    return ((state << 1) & 0xFF) | feedback

def lfsr8_sequence(seed, n):
    """Generate n values from an 8-bit LFSR with given seed."""
    vals, state = [], seed
    for _ in range(n):
        state = lfsr8_next(state)
        vals.append(state)
    return vals

# ─── Noise generation (sum 4 LFSR outputs, rescale) ───────────────────
def gaussian_approx_noise(lfsr_states):
    """
    Sum 4 LFSR bytes, subtract mean (4*127=508), scale to Q8.8.
    Mirrors the hardware adder tree.
    Returns (noise_fixed, updated_lfsr_states)
    """
    total = 0
    for i in range(4):
        lfsr_states[i] = lfsr8_next(lfsr_states[i])
        total += lfsr_states[i]
    # center around 0, scale down to reasonable noise magnitude
    noise_float = (total - 508) / 256.0   # tune divisor to set Q (noise power)
    return to_fixed(noise_float), lfsr_states

print("Fixed-point config loaded. SCALE =", SCALE)
print("Q8.8 range: ", to_float(MIN_VAL), "to", to_float(MAX_VAL))

# Process noise std dev — how much the true state drifts per step
Q_FLOAT = 0.5
Q_FIXED = to_fixed(Q_FLOAT)   # = 128 in Q8.8

# Measurement noise std dev — how noisy the sensor is
R_FLOAT = 1.0
R_FIXED = to_fixed(R_FLOAT)   # = 256 in Q8.8

# Resampling threshold (rectangular likelihood half-width)
DELTA_FLOAT = 1.5
DELTA_FIXED = to_fixed(DELTA_FLOAT)   # = 384 in Q8.8

print(f"Q = {Q_FIXED} ({to_float(Q_FIXED):.3f})")
print(f"R = {R_FIXED} ({to_float(R_FIXED):.3f})")
print(f"delta = {DELTA_FIXED} ({to_float(DELTA_FIXED):.3f})")