# ─── kalman_filter.py ─────────────────────────────────────────────────────────
# Scalar 1D Kalman Filter baseline for the particle filter project
#
# Three jobs:
#   1. Implement the scalar KF (predict + update loop)
#   2. Derive equivalent R from rectangular threshold delta
#   3. Generate ONE fixed sequence shared with the hardware sim
#
# Every number here uses the same Q_STD, DELTA, N_STEPS as particle_filter.py
# ──────────────────────────────────────────────────────────────────────────────

import numpy as np
import csv
import os

# ─── Parameters ───────────────────────────────────────────────────────────────
# These must match particle_filter.py exactly — change both or neither

Q_STD   = 0.5    # process noise std dev  (how much state drifts per step)
DELTA   = 1.5    # rectangular likelihood half-width (from particle filter)
N_STEPS = 500    # number of time steps
GT_SEED = 42     # ground truth RNG seed — NEVER change this after locking in

# Derived constants
Q = Q_STD ** 2   # process noise variance = std dev squared = 0.25

# ─── Core Functions ───────────────────────────────────────────────────────────

def compute_equivalent_R(delta):
    """
    Derive equivalent Gaussian R from rectangular likelihood threshold delta.

    Your particle filter uses a rectangular (uniform) likelihood window:
        weight = 1  if  |z - x| <= delta
        weight = 0  otherwise

    This is equivalent to assuming measurement noise is Uniform[-delta, +delta].
    The variance of Uniform[-delta, +delta] is delta^2 / 3.

    To compare the KF fairly, use this R so both filters assume
    the same effective measurement uncertainty.

    Args:
        delta : rectangular likelihood half-width (float)

    Returns:
        R : equivalent Gaussian measurement noise variance (float)
    """
    return (delta ** 2) / 3


def compute_steady_state(Q, R):
    """
    Analytically compute steady-state variance P_inf and gain K_inf.

    Derived by setting P_k = P_{k-1} = P in the Riccati equation and solving
    the resulting quadratic. In 1D this has a closed-form solution.

    Why this matters for hardware:
        Once K converges to K_inf it stops changing every cycle.
        You can precompute K_inf here and hardcode it in Verilog as a
        fixed-point constant — completely eliminating the need for a
        hardware divider.

    Args:
        Q : process noise variance
        R : measurement noise variance

    Returns:
        P_inf : steady-state error variance
        K_inf : steady-state Kalman gain (between 0 and 1)
    """
    P_inf = Q / 2 + np.sqrt((Q / 2) ** 2 + Q * R)
    K_inf = P_inf / (P_inf + R)
    return P_inf, K_inf


def run_kalman_filter(measurements, Q, R, x0=0.0, P0=1.0):
    """
    Run the scalar 1D Kalman filter on a measurement sequence.

    System model:
        x_k = x_{k-1} + w_k,    w_k ~ N(0, Q)   process noise
        z_k = x_k     + v_k,    v_k ~ N(0, R)   measurement noise

    Two steps every cycle:
        PREDICT:  x_pred = x_hat            (random walk, no drift)
                  P_pred = P + Q            (uncertainty grows)
        UPDATE:   K      = P_pred / (P_pred + R)          (Kalman gain)
                  x_hat  = x_pred + K * (z_k - x_pred)   (correct estimate)
                  P      = (1 - K) * P_pred               (uncertainty shrinks)

    Args:
        measurements : np.array of noisy sensor readings z_k, shape (n,)
        Q            : process noise variance
        R            : measurement noise variance
        x0           : initial state estimate (default 0.0)
        P0           : initial uncertainty    (default 1.0)

    Returns:
        estimates : np.array of x̂_k, shape (n,)
        P_log     : np.array of P_k  — variance at each step
        K_log     : np.array of K_k  — Kalman gain at each step
    """
    n         = len(measurements)
    estimates = np.zeros(n)
    P_log     = np.zeros(n)
    K_log     = np.zeros(n)

    x_hat = x0
    P     = P0

    for k in range(n):

        # ── PREDICT ───────────────────────────────────────────────
        x_pred = x_hat
        P_pred = P + Q

        # ── UPDATE ────────────────────────────────────────────────
        innovation = measurements[k] - x_pred
        K          = P_pred / (P_pred + R)
        x_hat      = x_pred + K * innovation
        P          = (1 - K) * P_pred

        estimates[k] = x_hat
        P_log[k]     = P
        K_log[k]     = K

    return estimates, P_log, K_log


# ─── Shared Sequence Generator ────────────────────────────────────────────────

def generate_shared_sequence(n_steps, q_std, r_std, seed):
    """
    Generate ONE fixed ground truth + measurement sequence.

    This is the sequence fed identically to:
        - the Kalman filter (here, in Python)
        - the hardware particle filter sim (via testbench CSV input)

    Using the same seed every time means the comparison is apples-to-apples:
    both filters see exactly the same noise realizations.
    Never change GT_SEED once you've locked it in.

    Args:
        n_steps : number of time steps
        q_std   : process noise std dev
        r_std   : measurement noise std dev (sqrt(R_equivalent))
        seed    : RNG seed — use GT_SEED constant

    Returns:
        true_state   : np.array shape (n_steps,) — hidden truth
        measurements : np.array shape (n_steps,) — noisy sensor readings
    """
    rng          = np.random.default_rng(seed)
    true_state   = np.zeros(n_steps)
    measurements = np.zeros(n_steps)

    for k in range(1, n_steps):
        true_state[k]   = true_state[k-1] + rng.normal(0, q_std)
        measurements[k] = true_state[k]   + rng.normal(0, r_std)

    return true_state, measurements


def save_sequence_to_csv(true_state, measurements, path="shared_sequence.csv"):
    """
    Save the shared sequence to CSV so the Verilog testbench can load it.

    The testbench reads this file and feeds measurements[k] into the
    hardware filter at each step — ensuring hardware and Python see
    identical inputs.

    CSV format:
        step, true_state, measurement
        0,    0.0000,     0.0312
        1,    0.4721,     1.2043
        ...
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "true_state", "measurement"])
        for k in range(len(true_state)):
            writer.writerow([k,
                             round(float(true_state[k]),   6),
                             round(float(measurements[k]), 6)])
    print(f"Saved shared sequence to {path}  ({len(true_state)} steps)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── step 1: derive R ──────────────────────────────────────────
    R = compute_equivalent_R(DELTA)

    print("=" * 55)
    print("Parameters")
    print("=" * 55)
    print(f"  Q_STD   = {Q_STD}    → Q = {Q:.4f}")
    print(f"  DELTA   = {DELTA}    → R = delta^2/3 = {R:.4f}")
    print(f"  N_STEPS = {N_STEPS}")
    print(f"  GT_SEED = {GT_SEED}  (never change this)")

    # ── step 2: steady-state analysis ────────────────────────────
    P_inf, K_inf = compute_steady_state(Q, R)

    print()
    print("=" * 55)
    print("Steady-State Analysis")
    print("=" * 55)
    print(f"  P_inf  = {P_inf:.6f}  (converged variance)")
    print(f"  K_inf  = {K_inf:.6f}  (converged Kalman gain)")
    print(f"  K_inf in Q8.8 = {int(round(K_inf * 256))}  "
          f"← hardcode this in Verilog")

    # ── step 3: generate fixed shared sequence ────────────────────
    print()
    print("=" * 55)
    print("Generating Shared Sequence")
    print("=" * 55)

    # R_equivalent was derived from delta, but the actual measurement
    # noise std dev used to generate the sequence is sqrt(R)
    R_std        = np.sqrt(R)
    true_state, measurements = generate_shared_sequence(
        N_STEPS, Q_STD, R_std, seed=GT_SEED)

    save_sequence_to_csv(true_state, measurements)

    print(f"  True state   — min: {true_state.min():+.3f}  "
          f"max: {true_state.max():+.3f}")
    print(f"  Measurements — min: {measurements.min():+.3f}  "
          f"max: {measurements.max():+.3f}")

    # ── step 4: run the Kalman filter ─────────────────────────────
    print()
    print("=" * 55)
    print("Running Kalman Filter")
    print("=" * 55)

    kf_estimates, P_log, K_log = run_kalman_filter(
        measurements, Q=Q, R=R)

    kf_mse = np.mean((kf_estimates - true_state) ** 2)

    print(f"  Kalman MSE      = {kf_mse:.6f}  ← theoretical lower bound")
    print(f"  K at step   0   = {K_log[0]:.6f}  (not yet converged)")
    print(f"  K at step  50   = {K_log[50]:.6f}")
    print(f"  K at step 499   = {K_log[-1]:.6f}  (should ≈ K_inf={K_inf:.6f})")
    print(f"  P at step 499   = {P_log[-1]:.6f}  (should ≈ P_inf={P_inf:.6f})")

    # verify K actually converged
    converged = abs(K_log[-1] - K_inf) < 1e-6
    print(f"  K converged?    = {converged}")

    # ── step 5: save KF results to CSV for plotting ───────────────
    with open("kalman_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "true_state", "measurement",
                         "kf_estimate", "P", "K"])
        for k in range(N_STEPS):
            writer.writerow([k,
                             round(float(true_state[k]),    6),
                             round(float(measurements[k]),  6),
                             round(float(kf_estimates[k]),  6),
                             round(float(P_log[k]),         6),
                             round(float(K_log[k]),         6)])
    print()
    print("Saved Kalman results to kalman_results.csv")

    # ── step 6: sanity check printout ─────────────────────────────
    print()
    print("=" * 55)
    print("First 5 Steps (sanity check)")
    print("=" * 55)
    print(f"  {'step':>4}  {'true':>8}  {'meas':>8}  "
          f"{'kf_est':>8}  {'K':>6}  {'P':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*8}")
    for k in range(5):
        print(f"  {k:>4}  {true_state[k]:>+8.4f}  "
              f"{measurements[k]:>+8.4f}  "
              f"{kf_estimates[k]:>+8.4f}  "
              f"{K_log[k]:>6.4f}  "
              f"{P_log[k]:>8.6f}")