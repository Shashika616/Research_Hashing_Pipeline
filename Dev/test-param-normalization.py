import os
import hmac
import struct
import math
import numpy as np
import matplotlib.pyplot as plt
from hashlib import sha256
from typing import Tuple

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

# --- Safe float from bytes ---
def safe_float_from_bytes(b: bytes, precision: int = 6) -> float:
    if len(b) < 4: b = b.ljust(4, b'\x00')
    bits = int.from_bytes(b, byteorder='big')
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0 or exponent == 255: exponent = 127
    if mantissa == 0: mantissa = 1
    safe_bits = (sign << 31) | (exponent << 23) | mantissa
    val = struct.unpack('>f', safe_bits.to_bytes(4,'big'))[0]
    val = max(min(val, 1e6), -1e6)
    if abs(val) < 1e-6: val = 1e-6 if val >= 0 else -1e-6
    return round(val, precision)

# --- Parameter range generation ---
def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
    """
    Generates a user-specific safe range for parameters p1/p2 with debug info.
    """
    def extract_strong_float(hmac_bytes: bytes) -> float:
       
        base_val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
       
        extra_bits = int.from_bytes(hmac_bytes[4:8], byteorder='big')
        
        frac_rand = (extra_bits % 10000) / 10000.0
        
        combined = abs(base_val) * (1 + frac_rand)
        
        scaled = (math.log10(combined + 1e-6) + 6) % 3.0
        res = round(10 ** scaled, 6)
       
        return res

    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_min', sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_max', sha256).digest()
   

    min_val = extract_strong_float(hmac_min)
    max_val = extract_strong_float(hmac_max)
    

    if max_val < min_val:
        min_val, max_val = max_val, min_val
        

    if abs(max_val - min_val) < 5.0:
        sep_hmac = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_sep', sha256).digest()
        
        separation = (int.from_bytes(sep_hmac[:2], 'big') % 50) / 10.0
        
        max_val = round(min_val + 5.0 + separation, 6)
       

    return round(min_val, 6), round(max_val, 6)

# --- Parameter normalization ---
def normalize_param(value: float, param_min: float, param_max: float, method: str = "linear") -> float:
    x = np.array([value], dtype=np.float64)
    def linear_norm(x): return param_min + (abs(x[0]) % 1.0) * (param_max - param_min)
    def log_norm(x):
        safe_val = np.clip(np.abs(x[0]), 1e-30, 1e30)
        log_val = np.log10(safe_val)
        return param_min + (log_val % 1.0) * (param_max - param_min)
    def clipped_norm(x):
        clipped = np.clip(x[0], -1.0, 1.0)
        return param_min + (clipped + 1.0)/2 * (param_max - param_min)
    def tanh_norm(x):
        squashed = np.tanh(x[0] / 1e10)
        return param_min + (squashed + 1)/2 * (param_max - param_min)
    methods = {"linear": linear_norm, "log": log_norm, "clipped": clipped_norm, "tanh": tanh_norm}
    if method not in methods: raise ValueError(f"Unknown normalization method: {method}")
    return round(methods[method](x), 6)

# --- Demo / Comparison ---
if __name__ == "__main__":
    password = "abc"
    salt1 = os.urandom(8)
    salt2 = os.urandom(8)
    chars = list(password)
    normalization_methods = ["linear", "log", "clipped", "tanh"]

    # Storage for plotting
    plot_data = {method: {"p1": [], "p2": []} for method in normalization_methods}

    for idx, c in enumerate(chars):
        print(f"Character '{c}':")
        # Generate ranges
        p1_min, p1_max = generate_user_param_range(salt1, salt2, b'p1')
        p2_min, p2_max = generate_user_param_range(salt1, salt2, b'p2')
        # Raw values
        raw_p1 = safe_float_from_bytes(os.urandom(4))
        raw_p2 = safe_float_from_bytes(os.urandom(4))
        print(f"  raw_p1={raw_p1}, raw_p2={raw_p2}")
        # Apply all normalization methods
        for method in normalization_methods:
            norm_p1 = normalize_param(raw_p1, p1_min, p1_max, method)
            norm_p2 = normalize_param(raw_p2, p2_min, p2_max, method)
            print(f"    [{method}] p1={norm_p1}, p2={norm_p2}")
            plot_data[method]["p1"].append(norm_p1)
            plot_data[method]["p2"].append(norm_p2)

    # --- Plotting ---
    plt.figure(figsize=(12,5))
    for i, method in enumerate(normalization_methods):
        plt.scatter([i]*len(plot_data[method]["p1"]), plot_data[method]["p1"], label=f"{method}-p1", marker='o', color=f"C{i}")
        plt.scatter([i]*len(plot_data[method]["p2"]), plot_data[method]["p2"], label=f"{method}-p2", marker='x', color=f"C{i}")
    plt.xticks(range(len(normalization_methods)), normalization_methods)
    plt.ylabel("Normalized values")
    plt.title("Comparison of Normalization Methods for p1 and p2")
    plt.legend()
    plt.grid(True)
    plt.show()
