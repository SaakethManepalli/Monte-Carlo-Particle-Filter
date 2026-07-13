# ─── lfsr_tb.py ───────────────────────────────────────────────────────────────
# Bit-accurate Python equivalent of lfsr_tb.v
# Runs the LFSR for 1000 samples and writes results.csv
# Mirrors: reset → release → for loop → $fdisplay → $fclose → $finish
# ──────────────────────────────────────────────────────────────────────────────

import csv
from lfsr8 import lfsr8_run

OUTPUT_FILE = "results.csv"
NUM_SAMPLES = 1000
SEED        = 0xAC      # matches rst → out <= 8'hAC in lfsr8.v

def run_testbench():
    # 1. collect samples (equivalent to reset release + for loop)
    samples = lfsr8_run(seed=SEED, n=NUM_SAMPLES)

    # 2. write CSV (equivalent to $fopen + $fdisplay loop + $fclose)
    with open(OUTPUT_FILE, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample", "value"])   # header row
        for sample_index, value in samples:
            writer.writerow([sample_index, value])

    # 3. done (equivalent to $display + $finish)
    print(f"Wrote {NUM_SAMPLES} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_testbench()