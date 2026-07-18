# ─── kalman_baseline.py ───────────────────────────────────────────────────────
# Scalar random-walk Kalman filter + matching trajectory generator.
#
# State model:   x_k = x_{k-1} + w_k,      w_k ~ N(0, Q)
# Measurement:   z_k = x_k + v_k,          v_k ~ N(0, R)
#
# This is the exact (optimal) filter for the same generative model the
# particle filter is estimating, so it's the accuracy baseline analysis.py
# compares the particle filter against.
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np

# noise assumptions shared with the particle filter (see bit_accurate_prototype.py)
Q_STD = 0.5   # process noise std dev (state drift per step)
R_STD = 1.0   # measurement noise std dev


def generate_trajectory(n_steps, x0=0.0, q_std=Q_STD, r_std=R_STD, seed=None):
    """Simulate a scalar random walk and noisy measurements of it."""
    rng = np.random.default_rng(seed)
    true_states  = np.empty(n_steps)
    measurements = np.empty(n_steps)

    x = x0
    for k in range(n_steps):
        x += rng.normal(0.0, q_std)
        z = x + rng.normal(0.0, r_std)
        true_states[k]  = x
        measurements[k] = z

    return true_states, measurements


def kalman_filter(measurements, x0=0.0, p0=1.0, q_std=Q_STD, r_std=R_STD):
    """Exact scalar Kalman filter for the random-walk model above.
    Returns (estimates, variances)."""
    q = q_std ** 2
    r = r_std ** 2

    n_steps = len(measurements)
    estimates = np.empty(n_steps)
    variances = np.empty(n_steps)

    x_est = x0
    p_est = p0

    for k in range(n_steps):
        # predict
        x_pred = x_est
        p_pred = p_est + q

        # update
        kalman_gain = p_pred / (p_pred + r)
        x_est = x_pred + kalman_gain * (measurements[k] - x_pred)
        p_est = (1.0 - kalman_gain) * p_pred

        estimates[k] = x_est
        variances[k] = p_est

    return estimates, variances


def mse(estimates, true_states):
    """Mean squared error between an estimate trajectory and ground truth."""
    return float(np.mean((np.asarray(estimates) - np.asarray(true_states)) ** 2))


if __name__ == "__main__":
    N_STEPS = 500
    true_states, measurements = generate_trajectory(N_STEPS, seed=0)
    estimates, variances = kalman_filter(measurements)

    print(f"Kalman baseline over {N_STEPS} steps")
    print(f"  measurement MSE: {mse(measurements, true_states):.4f}")
    print(f"  Kalman MSE:      {mse(estimates, true_states):.4f}")
