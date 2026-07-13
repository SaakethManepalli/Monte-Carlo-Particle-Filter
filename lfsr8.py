# ─── lfsr8.py ─────────────────────────────────────────────────────────────────
# Bit-accurate Python model of lfsr8.v
# Feedback polynomial: x^8 + x^6 + x^5 + x^4 + 1
# Taps at indices [7, 5, 4, 3] (0-indexed from LSB)
# Period: 2^8 - 1 = 255 (cycles through all non-zero states)
# ──────────────────────────────────────────────────────────────────────────────

SEED = 0xAC  # default non-zero seed, matches lfsr8.v

def lfsr8_next(state):
    """
    Compute one clock cycle of the 8-bit LFSR.
    Mirrors the Verilog line:
        out <= {out[6:0], out[7] ^ out[5] ^ out[4] ^ out[3]};
    """
    # extract the four tap bits (matching Verilog indices)
    bit7 = (state >> 7) & 1
    bit5 = (state >> 5) & 1
    bit4 = (state >> 4) & 1
    bit3 = (state >> 3) & 1

    feedback = bit7 ^ bit5 ^ bit4 ^ bit3

    # shift left by 1, mask to 8 bits, feed new bit into LSB
    next_state = ((state << 1) & 0xFF) | feedback

    return next_state


def lfsr8_run(seed=SEED, n=1000):
    """
    Run the LFSR for n cycles starting from seed.
    Mirrors the reset + for loop in lfsr_tb.v.
    Returns a list of (sample_index, value) tuples.
    """
    state   = seed
    samples = []

    for i in range(n):
        state = lfsr8_next(state)
        samples.append((i, state))

    return samples