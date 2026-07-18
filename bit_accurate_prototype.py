# ─── bit_accurate_prototype.py ────────────────────────────────────────────────
# Bit-accurate model of the particle-filter pipeline: verification oracle for
# the Verilog build. Fixed-point format locked in Phase 0: Q8.8 signed 16-bit
# (1 sign + 7 int + 8 frac bits, range -128..+127.996, step 1/256).
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np

from kalman_baseline import generate_trajectory, kalman_filter, mse

FRAC_BITS  = 8
SCALE      = 1 << FRAC_BITS
DATA_WIDTH = 16
MAX_VAL    = (1 << (DATA_WIDTH - 1)) - 1
MIN_VAL    = -(1 << (DATA_WIDTH - 1))
M          = 16   # particle count


def to_fixed(x):
    return int(np.clip(round(x * SCALE), MIN_VAL, MAX_VAL))


def to_float(fx):
    return fx / SCALE


def fixed_add(a, b):
    return int(np.clip(a + b, MIN_VAL, MAX_VAL))


def fixed_sub(a, b):
    return int(np.clip(a - b, MIN_VAL, MAX_VAL))


# ─── LFSR bank (noise source) ──────────────────────────────────────────────
# 8-bit maximal-length LFSR, taps [7,5,4,3], bit-identical to lfsr8.v.
# Four of these summed and rescaled approximate Gaussian noise in hardware.

DEFAULT_SEEDS = [0xAC, 0x37, 0x5F, 0xC1]


def lfsr8_next(state):
    feedback = ((state >> 7) ^ (state >> 5) ^ (state >> 4) ^ (state >> 3)) & 1
    return ((state << 1) & 0xFF) | feedback


def make_lfsr_bank(seeds=None):
    seeds = seeds or DEFAULT_SEEDS
    assert len(seeds) == 4 and all(s != 0 for s in seeds)
    return list(seeds)


def noise_sample(lfsr_bank):
    total = 0
    for i in range(4):
        lfsr_bank[i] = lfsr8_next(lfsr_bank[i])
        total += lfsr_bank[i]
    centered = total - 510        # mean of 4 uniform[1,255] ~= 510
    return to_fixed(centered / 64.0)


# ─── Propagate / Weight ────────────────────────────────────────────────────
def propagate(particle, noise):
    return fixed_add(particle, noise)


def weight(z, particle, delta):
    """Rectangular likelihood: pass (1) if |z - particle| < delta, else fail (0)."""
    return 1 if abs(fixed_sub(z, particle)) < delta else 0


# ─── Resample FSM (Accumulate / Select / Write) ────────────────────────────
def resample_accumulate(pass_flags):
    """ACCUMULATE: running pass-count table c_sum[i] = passes among [0..i]."""
    c_sum = [0] * M
    running = 0
    for i in range(M):
        running += pass_flags[i]
        c_sum[i] = running
    return c_sum, running


def resample_select(out_j, c_sum, n_pass):
    """SELECT: source index for slot out_j via cross-multiplication (no divider):
    first i where c_sum[i] * M > out_j * n_pass. n_pass == 0 keeps particles in place."""
    if n_pass == 0:
        return out_j
    target = out_j * n_pass
    for i in range(M):
        if c_sum[i] * M > target:
            return i
    return M - 1


def resample(pass_flags, particles):
    """Accumulate, then Select+Write every output slot."""
    c_sum, n_pass = resample_accumulate(pass_flags)
    new_particles = [particles[resample_select(j, c_sum, n_pass)] for j in range(M)]
    return new_particles, n_pass


# ─── Estimator (adder tree) ────────────────────────────────────────────────
def estimator(particles):
    """Binary adder tree over M=16 particles, mean via arithmetic shift (no divider)."""
    assert M == 16
    level1 = [particles[2 * i] + particles[2 * i + 1] for i in range(8)]
    level2 = [level1[2 * i] + level1[2 * i + 1] for i in range(4)]
    level3 = [level2[2 * i] + level2[2 * i + 1] for i in range(2)]
    total  = level3[0] + level3[1]
    return int(np.clip(total >> 4, MIN_VAL, MAX_VAL))   # >> floors, matches Verilog >>>


# ─── Full pipeline ─────────────────────────────────────────────────────────
def run_particle_filter(n_steps, delta_float=1.5, seed=0):
    true_states, measurements = generate_trajectory(n_steps, seed=seed)
    delta_fixed = to_fixed(delta_float)

    lfsr_bank = make_lfsr_bank()
    particles = [0] * M   # prior centered at 0, matches kalman_baseline x0=0.0
    estimates = np.empty(n_steps)

    for k in range(n_steps):
        z_fixed = to_fixed(measurements[k])
        particles = [propagate(p, noise_sample(lfsr_bank)) for p in particles]
        pass_flags = [weight(z_fixed, p, delta_fixed) for p in particles]
        particles, n_pass = resample(pass_flags, particles)
        estimates[k] = to_float(estimator(particles))

    return true_states, measurements, estimates


if __name__ == "__main__":
    N_STEPS = 20

    true_states, measurements, pf_estimates = run_particle_filter(N_STEPS, seed=0)
    kf_estimates, _ = kalman_filter(measurements)

    print(f"{'step':>4}  {'true':>8}  {'meas':>8}  {'PF est':>8}  {'KF est':>8}")
    for k in range(N_STEPS):
        print(f"{k:>4}  {true_states[k]:>8.3f}  {measurements[k]:>8.3f}  "
              f"{pf_estimates[k]:>8.3f}  {kf_estimates[k]:>8.3f}")

    print()
    print(f"measurement MSE: {mse(measurements, true_states):.4f}")
    print(f"PF MSE:          {mse(pf_estimates, true_states):.4f}")
    print(f"KF MSE:          {mse(kf_estimates, true_states):.4f}")
