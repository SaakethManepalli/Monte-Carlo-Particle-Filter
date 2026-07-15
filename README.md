# 1D Particle Filter in Hardware

A sequential Monte Carlo particle filter implemented in Verilog, tracking a scalar
random-walk state from noisy measurements. Sixteen fixed-point particles are
propagated, weighted, and resampled each clock cycle; the running state estimate is
logged to CSV and benchmarked against an exact Kalman filter baseline in Python.

## What this project does

A particle filter estimates a hidden quantity you can't observe directly by tracking
many weighted guesses ("particles") at once and letting noisy measurements decide which
guesses survive. Here the hidden quantity is a 1D position drifting under random noise;
the filter estimates where it really is using only unreliable readings.

The design is structured as a **coprocessor**: the FPGA performs the repetitive
estimation work (propagate, weight, resample, average), while a host would handle
measurement generation and I/O in a full system.

## Architecture

The filter uses a **sequential, one-particle-per-cycle datapath** — a single shared
pipeline cycles through all 16 particles rather than replicating hardware 16 times,
keeping the design small.

| Stage | Function 
|-------|---------
| LFSR bank | Four independent 8-bit maximal-length LFSRs supplying noise 
| Particle register file | Sixteen 16-bit fixed-point registers holding particle positions 
| Propagate | Adds summed-LFSR (approx. Gaussian) noise to each particle 
| Weight | Rectangular likelihood: pass if `|z − x| < δ`, else fail 
| Resample | Systematic resampling FSM (Accumulate / Select / Write) 
| Estimator | Adder tree computing the particle mean 

## Repository structure

```
verilog/
  lfsr8.v                 8-bit maximal-length LFSR (instantiated x4)
  particle_regfile.v      16 x 16-bit particle register file
  propagate.v             noise-add stage
  weight.v                rectangular-likelihood comparator
  resample_fsm.v          Accumulate / Select / Write state machine
  estimator.v             adder-tree mean
  particle_filter_top.v   top-level wiring + master FSM
  particle_filter_tb.v    testbench, writes results.csv

python/
  bit_accurate_prototype.py   fixed-point reference model (verification oracle)
  kalman_baseline.py          scalar Kalman filter + trajectory generator
  analysis.py                 plots: PF vs KF vs truth, MSE vs M

data/
  results.csv             simulation output (step, true, measurement, estimate, Neff)
  *.png                   generated plots
```

## Workflow

1. **Bit-accurate Python prototype first.** Every stage is prototyped in Python using
   integer arithmetic that mirrors the Q10.5 fixed-point format exactly (same width, same
   truncation, same saturation). This is the verification oracle — Verilog output is
   diffed against it value-by-value.
2. **Verilog build**, module by module, each tested against its Python counterpart.
3. **Integration** — wire the pipeline, confirm it tracks the true state.
4. **Data collection** — run 500 steps at M=16 for the main comparison, sweep M for the
   convergence curve.

## Running it

**Simulation (Vivado):** run the testbench `particle_filter_tb.v` as a behavioral
simulation. It writes `results.csv` into the xsim run directory.

**Analysis (Python):**
```
pip install -r requirements.txt
```
