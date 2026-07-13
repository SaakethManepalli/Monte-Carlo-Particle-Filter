# ─── particle_filter.py ───────────────────────────────────────────────────────
import numpy as np
import csv

# ─── Fixed-Point Configuration ────────────────────────────────────────────────
# Format: Q8.8 signed 16-bit
# 1 sign bit + 7 integer bits + 8 fractional bits
# Range:      -128.0 to +127.996
# Resolution: 1/256 ≈ 0.0039

FRAC_BITS  = 8
SCALE      = 1 << FRAC_BITS        # 256
MAX_VAL    = (1 << 15) - 1         #  32767  (+127.996 in Q8.8)
MIN_VAL    = -(1 << 15)            # -32768  (-128.0   in Q8.8)

# Accumulator for adder tree: Q8.8 but 20-bit wide
# Handles up to M=32 particles without overflow
# 32 × 32767 = 1,048,544 which fits in 20 bits (max 524,287 signed → use 21 to be safe)
ACC_MAX    = (1 << 20) - 1
ACC_MIN    = -(1 << 20)


def to_fixed(x):
    """
    Convert a Python float to Q8.8 fixed-point integer.
    Mirrors what the Verilog register stores.
    Saturates at boundaries instead of overflowing.
    """
    raw = int(round(x * SCALE))
    return int(np.clip(raw, MIN_VAL, MAX_VAL))


def to_float(fx):
    """
    Convert a Q8.8 fixed-point integer back to Python float.
    Use this only for analysis and printing — never inside the filter loop.
    """
    return fx / SCALE


def fixed_add(a, b):
    """
    Saturating 16-bit signed fixed-point addition.
    Mirrors a Verilog adder with overflow protection.
    In hardware this would be:
        wire signed [15:0] result = a + b;  (with saturation logic)
    """
    result = a + b
    return int(np.clip(result, MIN_VAL, MAX_VAL))


def fixed_sub(a, b):
    """
    Saturating 16-bit signed fixed-point subtraction.
    Used in the weight stage: |z_k - x_i|
    """
    result = a - b
    return int(np.clip(result, MIN_VAL, MAX_VAL))


def sign_extend_to_acc(val):
    """
    Sign-extend a 16-bit Q8.8 value to 20-bit accumulator width.
    Mirrors Verilog: {{4{particles[i][15]}}, particles[i]}
    Critical: without this, negative particles corrupt the mean.
    """
    if val & (1 << 15):          # if sign bit is set (negative number)
        return val | (0xF << 16) # extend sign bits into upper 4 bits
    return val

# ─── LFSR Bank ────────────────────────────────────────────────────────────────

# Four seeds — must all be non-zero, must all be different
# These match what you'll hardcode into lfsr8.v instances in Verilog
DEFAULT_SEEDS = [0xAC, 0x37, 0x5F, 0xC1]


def lfsr8_next(state):
    """
    One clock cycle of 8-bit LFSR.
    Identical to lfsr8.py — copy of the same function.
    Taps: [7, 5, 4, 3]
    """
    bit7 = (state >> 7) & 1
    bit5 = (state >> 5) & 1
    bit4 = (state >> 4) & 1
    bit3 = (state >> 3) & 1
    feedback   = bit7 ^ bit5 ^ bit4 ^ bit3
    next_state = ((state << 1) & 0xFF) | feedback
    return next_state


def make_lfsr_bank(seeds=None):
    """
    Create a mutable list of 4 LFSR states.
    Call this at the start of each filter run.
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    assert len(seeds) == 4, "Need exactly 4 seeds"
    assert all(s != 0 for s in seeds), "All seeds must be non-zero"
    return list(seeds)


def noise_sample(lfsr_bank):
    """
    Generate one Gaussian-approximated noise sample.
    Advances all 4 LFSRs by one step, sums their outputs,
    centers around zero, and scales to Q8.8.

    Mirrors the hardware adder tree:
        sum 4 LFSR bytes → subtract mean (4 × 127.5 = 510) → scale

    Returns:
        noise_fixed  : Q8.8 signed integer  (goes into Verilog register)
        lfsr_bank    : updated bank (same list, mutated in place)
    """
    total = 0
    for i in range(4):
        lfsr_bank[i] = lfsr8_next(lfsr_bank[i])
        total += lfsr_bank[i]

    # center: mean of 4 uniform[1..255] ≈ 510
    centered = total - 510

    # scale down: divide by 64 to get noise in roughly ±2.0 float range
    # in hardware this is a right-shift by 6
    noise_float = centered / 64.0

    noise_fixed = to_fixed(noise_float)
    return noise_fixed, lfsr_bank