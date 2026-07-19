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
Verilog/
  lfsr8.v                 8-bit maximal-length LFSR (instantiated x4)
  particle_regfile.v      16 x 16-bit particle register file
  propagate.v             noise-add stage
  weight.v                rectangular-likelihood comparator
  resample_fsm.v          Accumulate / Select / Write state machine
  estimator.v             adder-tree mean
  particle_filter_top.v   top-level wiring + master FSM
  Testbench/
    particle_filter_tb.v  testbench, writes results.csv

bit_accurate_prototype.py   fixed-point reference model (verification oracle)
kalman_baseline.py          scalar Kalman filter + trajectory generator
analysis.py                 plots: PF vs KF vs truth, MSE vs M (in progress)
```

`lfsr8.py`, `lfsr_tb.py`, `LSFR.py`, `particle_filter.py`, and
`particle_filter_bitsim.py` are earlier standalone experiments kept around for
reference; `bit_accurate_prototype.py` is the current, canonical model.

## Workflow

1. **Bit-accurate Python prototype first.** Every stage is prototyped in Python using
   integer arithmetic that mirrors the Q8.8 fixed-point format exactly (same width, same
   truncation, same saturation). This is the verification oracle — Verilog output is
   diffed against it value-by-value.
2. **Verilog build**, module by module, each tested against its Python counterpart.
3. **Integration** — wire the pipeline, confirm it tracks the true state.
4. **Data collection** — run 500 steps at M=16 for the main comparison, sweep M for the
   convergence curve.

## Running it

**Python prototype:**
```
pip install -r requirements.txt
python3 bit_accurate_prototype.py
```
Runs the bit-accurate pipeline for 20 steps and prints true state vs. measurement vs.
particle-filter estimate vs. Kalman estimate, plus MSE for each. `kalman_baseline.py`
can also be run standalone (500-step Kalman-only MSE check).

**Simulation (Vivado):** run the testbench `Testbench/particle_filter_tb.v` as a
behavioral simulation. It writes `results.csv` into the xsim run directory.
