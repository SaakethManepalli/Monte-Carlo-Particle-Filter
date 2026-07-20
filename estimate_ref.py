"""
estimate() -- bit-accurate Python reference for estimator.v

Sum all M particles in an ACC_W-bit accumulator, then arithmetic-shift
right by log2(M) to get the mean. This is the Phase 1 ground truth the
Verilog adder tree must match exactly, including on edge cases.
"""

N_WORD  = 16
MAX_VAL = (1 << (N_WORD - 1)) - 1     #  32767
MIN_VAL = -(1 << (N_WORD - 1))        # -32768

M       = 32
ACC_W   = 20
GROWTH  = 5                            # log2(32) -- shift amount for the mean
ACC_MIN = -(1 << (ACC_W - 1))          # -524288
ACC_MAX = (1 << (ACC_W - 1)) - 1       #  524287


def estimate(particles):
    """
    Mean of M particles via a 20-bit adder tree -- bit-accurate reference
    for estimator.v.

      - each 16-bit signed particle is sign-extended to ACC_W bits before
        adding (mirrors the RTL's per-leaf sign extension)
      - all M particles summed in a single ACC_W-bit accumulator
      - the assert below is the overflow guard: it fires BEFORE a bad
        vector could ever reach hardware, rather than letting the
        accumulator wrap silently and export a broken golden vector
      - arithmetic right shift by log2(M) for the mean -- Python's `>>`
        on an int is already an arithmetic/floor shift, which matches
        Verilog's `>>>` on a signed value
    """
    assert len(particles) == M, f"expected {M} particles, got {len(particles)}"
    for p in particles:
        assert MIN_VAL <= p <= MAX_VAL, f"particle {p} out of 16-bit signed range"

    total = sum(particles)

    assert ACC_MIN <= total <= ACC_MAX, (
        f"estimator accumulator overflow: sum={total} exceeds {ACC_W}-bit "
        f"signed range [{ACC_MIN}, {ACC_MAX}] -- widen ACC_W or bound "
        f"particle divergence before this happens"
    )

    return total >> GROWTH
