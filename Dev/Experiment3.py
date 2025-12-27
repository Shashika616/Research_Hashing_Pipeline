import os
import hmac
import struct
import hashlib
import math
import numpy as np
from collections import Counter
import time
import tracemalloc
import matplotlib.pyplot as plt

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

# ----------------------------
# Utilities
# ----------------------------
def safe_float_from_bytes(b: bytes, precision: int = 6) -> float:
    bits = int.from_bytes(b, byteorder='big')
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0 or exponent == 255:
        exponent = 127
    if mantissa == 0:
        mantissa = 1
    safe_bits = (sign << 31) | (exponent << 23) | mantissa
    safe_bytes = safe_bits.to_bytes(4, byteorder='big')
    val = struct.unpack('>f', safe_bytes)[0]
    min_val, max_val = 1e-6, 1e6
    val = max(min(val, max_val), -max_val)
    if abs(val) < min_val:
        val = min_val if val >= 0 else -min_val
    return round(val, precision)

def entropy(data: bytes) -> float:
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

# ----------------------------
# Plot function
# ----------------------------
def apply_plot_function(plot_type: int, p1: float, p2: float, x: float) -> float:
    MAX_MAG = 1e12
    MIN_MAG = 1e-12
    def clamp(v):
        if math.isnan(v) or math.isinf(v):
            return 0.0
        if v > MAX_MAG: return MAX_MAG
        if v < -MAX_MAG: return -MAX_MAG
        if abs(v) < MIN_MAG: return MIN_MAG * (1 if v >= 0 else -1)
        return v
    try:
        if plot_type == 0:
            val = p1*x**8 - p2*x**6 + p1**2*x**5 + math.sin(p1*x**3) + math.exp(p2*x/(1+abs(x))) - p1*math.log(abs(x)+10)
        elif plot_type == 1:
            val = ((p1*x**12 - p2*x**9 + p1*x**4)/(1+math.exp(-p2*x)) + math.tan(math.sin(p1*x)))
        elif plot_type == 2:
            val = p1*x**10 + p2*x**7 - p1*x**3 + math.sin(x**2) + math.log(abs(p2*x)+5) - math.exp(-abs(x))
        elif plot_type == 3:
            val = p1*math.exp(math.sin(p2*x)) - p2*math.log(abs(x*p1)+2) + x**5 - x**4 + p1*x**2
        elif plot_type == 4:
            val = math.sin(p1*x**3) + math.cos(p2*x**2) + math.tan(p1*x/(1+abs(x)))
        elif plot_type == 5:
            val = p1*x**20 - p2*x**17 + p1*x**14 - p1*p2*x**10 + p2*x**8 + math.sin(x) + math.exp(-abs(p1*x))
        else:
            val = 0.0
        return clamp(val)
    except:
        return 0.0

# ----------------------------
# Map character to function & params
# ----------------------------
def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes, user_range=(0.1,1.0), param_range=(0.5,2.0)):
    char_bytes = char.encode('utf-8')
    hmac_type = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'type', hashlib.sha256).digest()
    plot_type = hmac_type[0] % 6

    hmac_p1 = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'p1', hashlib.sha256).digest()
    p1_raw = safe_float_from_bytes(hmac_p1[:4])
    p1 = param_range[0] + (param_range[1] - param_range[0]) * abs(p1_raw % 1)

    hmac_p2 = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'p2', hashlib.sha256).digest()
    p2_raw = safe_float_from_bytes(hmac_p2[:4])
    p2 = param_range[0] + (param_range[1] - param_range[0]) * abs(p2_raw % 1)

    hmac_x = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'x', hashlib.sha256).digest()
    x_raw = safe_float_from_bytes(hmac_x[:4])
    x = user_range[0] + (user_range[1] - user_range[0]) * abs(x_raw % 1)

    return plot_type, p1, p2, x

# ----------------------------
# Iterative plot transform with Δy visualization
# ----------------------------
def iterative_plot_transform(password: str, salt1: bytes, salt2: bytes, iterations: int = 2):
    current_input = password.encode()
    all_intermediates = []
    metrics_per_iteration = []
    delta_y_all = []  # ← store Δy per character for visualization

    for it in range(1, iterations + 1):
        combined_input = current_input + salt1 + salt2
        values = []

        delta = 1e-7
        delta_ys = []

        for b in combined_input:
            char = chr(b % 256)

            plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)

            y1 = apply_plot_function(plot_type, p1, p2, x)
            y2 = apply_plot_function(plot_type, p1, p2, x + delta)

            dy = abs(y2 - y1)
            delta_ys.append(dy)

            values.append(y1)

        delta_y_all.append(delta_ys)
        delta_y = max(delta_ys)

        binary_data = struct.pack(f'{len(values)}f', *values)
        all_intermediates.append(binary_data)

        tracemalloc.start()
        start_time = time.time()
        entropy_val = entropy(binary_data)
        _, mem_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_time = time.time() - start_time

        # Preimage multiplicity
        first_char = chr(combined_input[0] % 256)
        plot_type, p1, p2, x = map_char_to_function_with_x(first_char, salt1, salt2)

        xs = np.linspace(x, x + 0.01, 1000)
        bucket = {}
        for val in xs:
            fp32 = struct.pack(">f", np.float32(apply_plot_function(plot_type, p1, p2, val)))
            bucket[fp32] = bucket.get(fp32, 0) + 1
        max_bucket = max(bucket.values())

        # FP32/FP64 collapse
        xs2 = np.linspace(x, x + 0.01, 1000)
        unique64 = set()
        unique32 = set()

        for val in xs2:
            y = apply_plot_function(plot_type, p1, p2, val)
            unique64.add(struct.pack(">d", float(y)))
            unique32.add(struct.pack(">f", np.float32(y)))

        collapse_factor = len(unique64) / len(unique32)

        metrics_per_iteration.append({
            "entropy": round(entropy_val, 4),
            "length": len(binary_data),
            "memory_kb": round(mem_peak / 1024, 2),
            "time": elapsed_time,
            "delta_y": delta_y,
            "preimage_mult": max_bucket,
            "fp32_fp64_collapse": round(collapse_factor, 2)
        })

        current_input = binary_data

    # -------- VISUALIZE Δy ----------
    for i, dy_list in enumerate(delta_y_all, 1):
        plt.figure(figsize=(10,4))
        plt.plot(dy_list)
        plt.title(f"Δy per character — Iteration {i}")
        plt.xlabel("Character index")
        plt.ylabel("Δy magnitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return current_input, all_intermediates, metrics_per_iteration


# ----------------------------
# Main terminal run
# ----------------------------
if __name__ == "__main__":
    passwords = ["abc", "password123", "12345678"]

    for pwd in passwords:
        salt1 = os.urandom(8)
        salt2 = os.urandom(8)

        final_binary, intermediates, metrics = iterative_plot_transform(
            pwd, salt1, salt2, iterations=2
        )

        print(f"\n=== Password: {pwd} ===")
        for i, (data, m) in enumerate(zip(intermediates, metrics), 1):
            floats = struct.unpack(f"{len(data)//4}f", data)
            hex_repr = data.hex()

            print(f"\n--- Iteration {i} ---")
            print("Intermediate floats:", floats)
            print("Intermediate hex:", hex_repr)
            print(
                f"Entropy={m['entropy']}, Length={m['length']}, Memory={m['memory_kb']} KB, "
                f"Time={m['time']:.6f}s, Δy={m['delta_y']:.6e}, Preimage Mult={m['preimage_mult']}, "
                f"FP32/FP64 Collapse={m['fp32_fp64_collapse']}"
            )
