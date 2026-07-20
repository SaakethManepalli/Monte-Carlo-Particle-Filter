"""
resample.py -- Phase 1 ground truth for the Resample FSM.

Mirrors the three states the Verilog FSM will implement:
  Accumulate -> prefix sum of the binary weights
  Select     -> systematic resampling: M evenly-spaced positions, walked
                with a single forward-moving pointer (never backtracks --
                this is what makes it a natural single-pass hardware sweep)
  Write      -> gather the new particle set from the selected indices

Also defines the zero-survivor policy explicitly, per Phase 0.
"""

from pylfsr import LFSR

# ── GLOBALS (matches the rest of the prototype) ────────────────────────────
N_WORD  = 16
N_FRAC  = 5
SCALE   = 1 << N_FRAC
MAX_VAL = (1 << (N_WORD - 1)) - 1
MIN_VAL = -(1 << (N_WORD - 1))
M       = 32

state = [0, 0, 0, 1, 0, 0, 0, 1]
fpoly = [8, 6, 5, 4]
L = LFSR(fpoly=fpoly, initstate=state)


def sat(x):
    return max(MIN_VAL, min(MAX_VAL, x))

def to_q105(x):
    return sat(int(x * SCALE))

def from_q105(fx):
    return fx / SCALE

def fx_add(a, b):
    return sat(a + b)

def fx_sub(a, b):
    return sat(a - b)

def run_LFSR(machine, n_bits=8):
    val = 0
    for i in range(8):
        machine.next()
        val = (val << 1) | int(machine.outbit)
    return val

def noise_sample(machine):
    total = sum(run_LFSR(machine) for _ in range(4))
    noise_float = (total - 510) / 256.0
    return to_q105(noise_float)

def propogate(particles, machine):
    return [fx_add(p, noise_sample(machine)) for p in particles]

DELTA = to_q105(1.5)

def weight(particles, z):
    weights = []
    for p in particles:
        diff = abs(fx_sub(z, p))
        weights.append(1 if diff <= DELTA else 0)
    return weights

ACC_W  = 20
GROWTH = 5
ACC_MIN = -(1 << (ACC_W - 1))
ACC_MAX = (1 << (ACC_W - 1)) - 1

def estimate(particles):
    assert len(particles) == M
    total = sum(particles)
    assert ACC_MIN <= total <= ACC_MAX, f"estimator overflow: sum={total}"
    return total >> GROWTH


# ── RESAMPLE FSM (Accumulate / Select / Write) ──────────────────────────────

def accumulate(weights):
    """Accumulate state: prefix sum of binary weights. Returns (cum_weight, total_weight)."""
    n = len(weights)
    cum_weight = [0] * n
    running = 0
    for i in range(n):
        running += weights[i]
        cum_weight[i] = running
    return cum_weight, running   # running == total_weight == cum_weight[-1]


def select(cum_weight, total_weight, n):
    """
    Select state: systematic resampling.
    target_j = floor(j * total_weight / n) for j = 0..n-1 (monotonic
    nondecreasing in j). Single forward-moving pointer walks cum_weight
    to find the first index whose cumulative weight exceeds target_j.

    n is fixed at M=32 (power of 2), so j*total_weight/n is a multiply
    followed by a fixed right-shift by log2(n) in hardware -- no general
    divider needed.
    """
    new_indices = []
    idx = 0
    for j in range(n):
        target = (j * total_weight) // n
        while cum_weight[idx] <= target:
            idx += 1
        new_indices.append(idx)
    return new_indices


def write(particles, new_indices):
    """Write state: gather the new particle set from selected indices."""
    return [particles[i] for i in new_indices]


def resample(particles, weights):
    """
    Full Accumulate -> Select -> Write pipeline.
    Returns (new_particles, new_indices, zero_survivors).

    Zero-survivor policy (Phase 0 decision): if no particle passed the
    weight threshold this step, hold the previous particle set unchanged
    rather than divide by zero or write garbage. The FSM should re-attempt
    propagate/weight next cycle rather than corrupt the register file.
    """
    n = len(particles)
    cum_weight, total_weight = accumulate(weights)

    if total_weight == 0:
        return list(particles), list(range(n)), True

    new_indices = select(cum_weight, total_weight, n)
    new_particles = write(particles, new_indices)
    return new_particles, new_indices, False


# ── SELF-TEST: zero-survivor edge case, isolated ────────────────────────────
def _test_zero_survivors():
    particles = [to_q105(v) for v in range(M)]     # arbitrary distinct values
    weights = [0] * M                               # nothing survives
    new_particles, new_indices, zero_flag = resample(particles, weights)
    assert zero_flag is True
    assert new_particles == particles, "zero-survivor case must hold previous set"
    print("Zero-survivor edge case: OK (held previous particle set unchanged)")


# ── INTEGRATION TEST: full pipeline over a handful of steps ────────────────
def run_pipeline(n_steps=15, true_x0=0.0, seed_offset=0):
    """
    Propagate -> Weight -> Resample -> Estimate, run against a synthetic
    true trajectory + noisy measurement, so the estimate can be sanity
    checked against ground truth -- same discipline as kalman_baseline.py.
    """
    import random
    rng = random.Random(42 + seed_offset)

    true_x = true_x0
    particles = [to_q105(true_x0)] * M   # start all particles at x0

    print(f"{'step':>4} {'true_x':>8} {'meas_z':>8} {'estimate':>9} {'survivors':>9}")
    for step in range(n_steps):
        # ground truth evolves independently of the filter's own noise source
        true_x += rng.gauss(0.0, 0.5)
        z_float = true_x + rng.gauss(0.0, 1.0)
        z = to_q105(z_float)

        particles = propogate(particles, L)
        weights = weight(particles, z)
        particles, indices, zero_flag = resample(particles, weights)
        est = estimate(particles)

        survivors = sum(weights)
        flag = " (ZERO SURVIVORS)" if zero_flag else ""
        print(f"{step:>4} {true_x:>8.3f} {z_float:>8.3f} {from_q105(est):>9.3f} {survivors:>9}{flag}")


if __name__ == "__main__":
    _test_zero_survivors()
    print()
    run_pipeline()
