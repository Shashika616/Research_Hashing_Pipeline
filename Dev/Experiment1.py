import numpy as np
import time
import struct
import hmac
import hashlib
from argon2 import PasswordHasher

# ================================================================
#  PART 0 — Your nonlinear plot-based function for a single char
# ================================================================
# ✔ You can replace this inner function with your real map_char_to_function_with_x()
# ---------------------------------------------------------------

def nonlinear_transform(char: str, x: float) -> float:
    """
    Example nonlinear function for experiments.
    Replace with your own plot-based character function!
    """
    v = ord(char)
    return np.sin(x * v) * np.cos(x * (v + 3)) + np.tan(x / (v + 1))


# ================================================================
#  PART 1 — CPU COST EXPERIMENT
# ================================================================
def experiment_cpu_cost(char="a", iterations=100000):
    start = time.time()
    s = 0
    for i in range(iterations):
        s += nonlinear_transform(char, i * 0.001)
    end = time.time()
    return end - start


# ================================================================
#  PART 2 — AVALANCHE EXPERIMENT
# ================================================================
def experiment_avalanche(char="a", x=12.345, delta=1e-7):
    y1 = nonlinear_transform(char, x)
    y2 = nonlinear_transform(char, x + delta)
    return abs(y2 - y1)


# ================================================================
#  PART 3 — PREIMAGE MULTIPLICITY EXPERIMENT
# ================================================================
def experiment_preimage_mult(char="a"):
    """
    Sweep many x values → count how many map to the same FP32 bucket.
    """
    xs = np.linspace(0, 50, 50000)
    bucket_counts = {}

    for x in xs:
        y = nonlinear_transform(char, x)
        # Quantize to float32 as in your pipeline
        y32 = np.float32(y)
        b = struct.pack(">f", y32)  # FP32 bin
        bucket_counts[b] = bucket_counts.get(b, 0) + 1

    # Find the largest bucket
    max_bucket = max(bucket_counts.values())
    return max_bucket, len(xs)


# ================================================================
#  PART 4 — IEEE 754 QUANTIZATION COLLAPSE EXPERIMENT
# ================================================================
def experiment_quantization_collapse(char="a"):
    xs = np.linspace(0, 100, 20000)
    unique_64 = set()
    unique_32 = set()

    for x in xs:
        y = nonlinear_transform(char, x)
        unique_64.add(struct.pack(">d", float(y)))    # full double precision
        unique_32.add(struct.pack(">f", np.float32(y)))  # float32 collapse

    return len(unique_64), len(unique_32)


# ================================================================
#  PART 5 — FULL PIPELINE COST MULTIPLIER
# ================================================================
def full_pipeline(password: str):
    # --------------------
    # STAGE 1 — Nonlinear + FP32
    # --------------------
    float_outputs = []
    for i, c in enumerate(password):
        y = nonlinear_transform(c, i * 0.123 + 1.234)
        y32 = struct.pack(">f", np.float32(y))
        float_outputs.append(y32)

    combined = b"".join(float_outputs)

    # --------------------
    # STAGE 2 — HMAC mixing
    # --------------------
    key = b"EXPERIMENT_SECRET"
    hmac_out = hmac.new(key, combined, hashlib.sha256).digest()

    # --------------------
    # STAGE 3 — Argon2
    # --------------------
    ph = PasswordHasher(time_cost=2, memory_cost=51200, parallelism=2)
    start = time.time()
    h = ph.hash(hmac_out.hex())
    end = time.time()

    return end - start


def experiment_pipeline_cost():
    base_cost = experiment_cpu_cost(iterations=1000) / 1000
    pipeline_cost = full_pipeline("abc")

    multiplier = pipeline_cost / base_cost
    return pipeline_cost, base_cost, multiplier


# ================================================================
#  RUN ALL EXPERIMENTS
# ================================================================
if __name__ == "__main__":
    print("=======================================================")
    print(" EXPERIMENT 1 — CPU COST")
    print("=======================================================")
    cpu_time = experiment_cpu_cost()
    print(f"Time for 100,000 nonlinear evaluations: {cpu_time:.4f}s")

    print("\n=======================================================")
    print(" EXPERIMENT 2 — AVALANCHE")
    print("=======================================================")
    diff = experiment_avalanche()
    print(f"|Δy| after Δx = 1e-7 → {diff}")

    print("\n=======================================================")
    print(" EXPERIMENT 3 — PREIMAGE MULTIPLICITY")
    print("=======================================================")
    max_bucket, total = experiment_preimage_mult()
    print(f"Total x values tested: {total}")
    print(f"Max # of x values mapping to SAME FP32 output: {max_bucket}")

    print("\n=======================================================")
    print(" EXPERIMENT 4 — IEEE 754 QUANTIZATION COLLAPSE")
    print("=======================================================")
    u64, u32 = experiment_quantization_collapse()
    print(f"Unique FP64 values: {u64}")
    print(f"Unique FP32 values: {u32}")
    print(f"Collapse factor (FP64→FP32): {u64 / u32:.2f}x reduction")

    print("\n=======================================================")
    print(" EXPERIMENT 5 — FULL PIPELINE COST MULTIPLIER")
    print("=======================================================")
    pipe, base, multi = experiment_pipeline_cost()
    print(f"Base nonlinear cost per eval: {base:.8f} s")
    print(f"Full pipeline cost: {pipe:.4f} s")
    print(f"Attacker cost multiplier: {multi:.2f}x")
