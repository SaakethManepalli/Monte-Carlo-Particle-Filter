import random
from estimate_ref import estimate, M, MIN_VAL, MAX_VAL, ACC_MIN, ACC_MAX

random.seed(1)

cases = []

# 1) all zero
cases.append([0] * M)

# 2) all same small positive / negative value
cases.append([100] * M)
cases.append([-100] * M)

# 3) single nonzero particle, rest zero -- isolates one leaf of the tree
c = [0] * M
c[0] = MAX_VAL
cases.append(c)
c = [0] * M
c[-1] = MIN_VAL
cases.append(c)

# 4) alternating MAX_VAL / MIN_VAL -- stresses sign extension hardest:
#    if a leaf's sign-extend is wrong, this sum will be wildly off instead
#    of the correct near-cancellation
cases.append([MAX_VAL if i % 2 == 0 else MIN_VAL for i in range(M)])

# 5) positive boundary: largest per-particle value that keeps the total
#    just inside the 20-bit accumulator's positive limit
p_pos = ACC_MAX // M          # 16383
cases.append([p_pos] * M)

# 6) negative boundary: exactly at the 20-bit accumulator's negative limit
p_neg = ACC_MIN // M          # -16384
cases.append([p_neg] * M)

# 7) random cases across the full 16-bit range, but only keep ones whose
#    sum doesn't trip the overflow assert (i.e. realistic operating points)
while len(cases) < 40:
    trial = [random.randint(MIN_VAL, MAX_VAL) for _ in range(M)]
    if ACC_MIN <= sum(trial) <= ACC_MAX:
        cases.append(trial)

# ── export ──────────────────────────────────────────────────────────────
with open("estimator_particles.hex", "w") as fp, open("estimator_expected.hex", "w") as fe:
    for particles in cases:
        mean = estimate(particles)          # bit-accurate reference, asserts on overflow
        for p in particles:
            fp.write(f"{p & 0xFFFF:04x}\n")
        fe.write(f"{mean & 0xFFFF:04x}\n")

print(f"Wrote {len(cases)} test cases ({len(cases)*M} particle values)")
print("First case means (raw Q10.5 int):", [estimate(c) for c in cases[:6]])
