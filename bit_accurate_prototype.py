# ─── bit_accurate_prototype.py ────────────────────────────────────────────────
# Bit-accurate model of the particle-filter pipeline: verification oracle for
# the Verilog build.
#
# Fixed-point format: Q10.5 signed 16-bit (1 sign + 10 int + 5 frac bits),
# matching LSFR.py / resample.py / estimate_ref.py -- and the Verilog already
# built and golden-vector-verified this session:
#   lfsr_core.v / lfsr_byte_gen.v   (64/64 match vs. this file's LFSR)
#   estimator.v                     (40/40 match vs. this file's estimator())
#   resample_fsm.v                  (16/16 match vs. this file's resample(),
#                                     including the zero-survivor case)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
from pylfsr import LFSR

from kalman_baseline import generate_trajectory, kalman_filter, mse

# ─── Fixed-point config (Q10.5) ─────────────────────────────────────────────
N_FRAC     = 5
SCALE      = 1 << N_FRAC                   # 32
DATA_WIDTH = 16
MAX_VAL    = (1 << (DATA_WIDTH - 1)) - 1   #  32767
MIN_VAL    = -(1 << (DATA_WIDTH - 1))      # -32768
M          = 32                             # particle count -- matches estimator.v / resample_fsm.v


def to_fixed(x):
    """Float -> Q10.5 raw integer. Truncates (int()), not banker's-rounds, and saturates
    -- exactly as a register would."""
    return int(np.clip(int(x * SCALE), MIN_VAL, MAX_VAL))


def to_float(fx):
    """Q10.5 raw integer -> float. For printing/analysis only."""
    return fx / SCALE


def fixed_add(a, b):
    return int(np.clip(a + b, MIN_VAL, MAX_VAL))


def fixed_sub(a, b):
    return int(np.clip(a - b, MIN_VAL, MAX_VAL))


# ─── LFSR noise source ──────────────────────────────────────────────────────
# ONE 8-bit maximal-length LFSR (pylfsr, fpoly=[8,6,5,4]), advanced 8 bits per
# byte -- bit-identical to lfsr_core.v / lfsr_byte_gen.v. noise_sample() draws
# 4 consecutive bytes from this single stream (not 4 separate LFSR instances).
_LFSR_INITSTATE = [0, 0, 0, 1, 0, 0, 0, 1]
_LFSR_FPOLY     = [8, 6, 5, 4]


def make_lfsr():
    """Fresh LFSR instance -- call once per filter run."""
    return LFSR(fpoly=_LFSR_FPOLY, initstate=list(_LFSR_INITSTATE))


def run_LFSR(machine):
    """Assemble one 8-bit byte from 8 consecutive LFSR outbits, MSB-first.
    Bit-identical to lfsr_byte_gen.v (verified 64/64 against this exact function)."""
    val = 0
    for _ in range(8):
        machine.next()
        val = (val << 1) | int(machine.outbit)
    return val


def noise_sample(machine):
    """Sum 4 consecutive LFSR bytes, center, scale to Q10.5."""
    total = sum(run_LFSR(machine) for _ in range(4))
    noise_float = (total - 510) / 256.0    # center + scale, ~±2
    return to_fixed(noise_float)


# ─── Propagate / Weight ─────────────────────────────────────────────────────
def propagate(particle, noise):
    return fixed_add(particle, noise)


DELTA = to_fixed(1.5)


def weight(z, particle, delta=DELTA):
    """Rectangular likelihood: pass (1) if |z - particle| <= delta, else fail (0)."""
    return 1 if abs(fixed_sub(z, particle)) <= delta else 0


# ─── Resample FSM (Accumulate / Select / Write) ────────────────────────────
def resample_accumulate(pass_flags):
    """ACCUMULATE: prefix sum of the binary pass flags."""
    n = len(pass_flags)
    cum_weight = [0] * n
    running = 0
    for i in range(n):
        running += pass_flags[i]
        cum_weight[i] = running
    return cum_weight, running


def resample_select(cum_weight, total_weight, n):
    """
    SELECT: systematic resampling, single forward-moving pointer (never
    backtracks -- target_j is monotonic nondecreasing in j, same as
    cum_weight is monotonic nondecreasing in index).
    n=M=32 is a power of 2, so target_j = j*total_weight/n is a multiply +
    fixed 5-bit right-shift in hardware, no divider needed.
    Bit-identical to resample_fsm.v's S_SELECT_CALC / S_SELECT_SRCH states.
    """
    new_indices = []
    idx = 0
    for j in range(n):
        target = (j * total_weight) // n
        while cum_weight[idx] <= target:
            idx += 1
        new_indices.append(idx)
    return new_indices


def resample(pass_flags, particles):
    """
    Accumulate -> Select -> Write. Returns (new_particles, zero_survivors).

    Zero-survivor policy: if nothing passed the weight threshold this step,
    hold the previous particle set unchanged rather than divide by zero or
    search a cum_weight array that never rises above 0 -- matches
    resample_fsm.v's S_HOLD state.
    """
    n = len(particles)
    cum_weight, total_weight = resample_accumulate(pass_flags)

    if total_weight == 0:
        return list(particles), True

    new_indices = resample_select(cum_weight, total_weight, n)
    new_particles = [particles[i] for i in new_indices]
    return new_particles, False


# ─── Estimator (adder tree) ─────────────────────────────────────────────────
ACC_W   = 20
GROWTH  = 5                        # log2(M=32)
ACC_MIN = -(1 << (ACC_W - 1))
ACC_MAX = (1 << (ACC_W - 1)) - 1


def estimator(particles):
    """
    Sum all M=32 particles in a 20-bit accumulator, arithmetic-shift right
    by log2(M)=5 for the mean. Bit-identical to estimator.v (verified 40/40,
    including the alternating-extremes sign-extension stress case).
    The assert is the overflow guard: it fires before a bad vector could
    ever reach hardware.
    """
    assert len(particles) == M, f"expected {M} particles, got {len(particles)}"
    for p in particles:
        assert MIN_VAL <= p <= MAX_VAL, f"particle {p} out of 16-bit signed range"

    total = sum(particles)
    assert ACC_MIN <= total <= ACC_MAX, (
        f"estimator accumulator overflow: sum={total} exceeds {ACC_W}-bit "
        f"signed range [{ACC_MIN}, {ACC_MAX}]"
    )
    return total >> GROWTH


# ─── Full pipeline ─────────────────────────────────────────────────────────
def run_particle_filter(n_steps, delta_float=1.5, seed=0):
    true_states, measurements = generate_trajectory(n_steps, seed=seed)
    delta_fixed = to_fixed(delta_float)

    machine   = make_lfsr()
    particles = [0] * M          # prior centered at 0, matches kalman_baseline x0=0.0
    estimates = np.empty(n_steps)
    zero_survivor_steps = []

    for k in range(n_steps):
        z_fixed    = to_fixed(measurements[k])
        particles  = [propagate(p, noise_sample(machine)) for p in particles]
        pass_flags = [weight(z_fixed, p, delta_fixed) for p in particles]
        particles, zero_flag = resample(pass_flags, particles)
        if zero_flag:
            zero_survivor_steps.append(k)
        estimates[k] = to_float(estimator(particles))

    return true_states, measurements, estimates, zero_survivor_steps


if __name__ == "__main__":
    N_STEPS = 300

    true_states, measurements, pf_estimates, zero_steps = run_particle_filter(N_STEPS, seed=0)
    kf_estimates, _ = kalman_filter(measurements)

    print(f"{'step':>4}  {'true':>8}  {'meas':>8}  {'PF est':>8}  {'KF est':>8}")
    for k in range(N_STEPS):
        flag = "  (ZERO SURVIVORS)" if k in zero_steps else ""
        print(f"{k:>4}  {true_states[k]:>8.3f}  {measurements[k]:>8.3f}  "
              f"{pf_estimates[k]:>8.3f}  {kf_estimates[k]:>8.3f}{flag}")

    print()
    print(f"measurement MSE: {mse(measurements, true_states):.4f}")
    print(f"PF MSE:          {mse(pf_estimates, true_states):.4f}")
    print(f"KF MSE:          {mse(kf_estimates, true_states):.4f}")
    if zero_steps:
        print(f"\nZero-survivor steps: {zero_steps}")