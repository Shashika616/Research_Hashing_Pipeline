#!/usr/bin/env python3
"""
plot_hash_benchmark_refined_plotly.py

Complete refined password hashing and benchmarking script with interactive Plotly plots.
Hover tooltips show full Plot-only binary output and Plot+Argon2 hash.
"""

import os
import csv
import time
import hmac
import math
import struct
import hashlib
import bcrypt
import tracemalloc
import numpy as np
from argon2 import PasswordHasher
from collections import Counter
from typing import Tuple
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

DEBUG = True  # Toggle verbose debug output
NORMALIZATION_MODE = "log"  # Options: "linear", "log", "clipped", "tanh"
SECRET_KEY = os.environ.get("PLOT_SECRET_KEY", "dev-key-for-testing").encode()

# ============================================================
# SAFE FLOAT EXTRACTION
# ============================================================

def safe_float_from_bytes(b: bytes, precision: int = 6) -> float:
    if len(b) < 4:
        b = b.ljust(4, b'\x00')
    bits = int.from_bytes(b[:4], 'big')
    sign = (bits >> 31) & 1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent in (0, 255):
        exponent = 127
    if mantissa == 0:
        mantissa = 1
    new_bits = (sign << 31) | (exponent << 23) | mantissa
    val = struct.unpack('>f', new_bits.to_bytes(4, 'big'))[0]
    val = max(min(val, 1e6), -1e6)
    if -1e-6 < val < 1e-6:
        val = 1e-6 if val >= 0 else -1e-6
    return round(val, precision)

# ============================================================
# RANGE AND PARAMETER GENERATION
# ============================================================

def extract_strong_float(hmac_digest: bytes) -> float:
    val = safe_float_from_bytes(hmac_digest[:4])
    exponent = math.log10(abs(val)) if val != 0 else 0.0
    scaled = 10 ** (exponent % 3.0)
    return round(scaled, 6)

def generate_user_range(salt1: bytes, salt2: bytes) -> Tuple[float, float]:
    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + b"user_x_range_min", hashlib.sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + b"user_x_range_max", hashlib.sha256).digest()
    x_min = extract_strong_float(hmac_min)
    x_max = extract_strong_float(hmac_max)
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if abs(x_max - x_min) < 1.0:
        adjustment = extract_strong_float(hmac.new(SECRET_KEY, salt1 + salt2 + b"adjust_range", hashlib.sha256).digest())
        x_max = round(x_min + abs(adjustment) + 1.0, 6)
    return (x_min, x_max)

def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + label + b"_min", hashlib.sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + label + b"_max", hashlib.sha256).digest()
    min_val = extract_strong_float(hmac_min)
    max_val = extract_strong_float(hmac_max)
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    if abs(max_val - min_val) < 5.0:
        max_val = min_val + 5.0
    return (round(min_val, 6), round(max_val, 6))

def normalize_param(value: float, param_min: float, param_max: float) -> float:
    frac = abs(value) - math.floor(abs(value))
    mapped = param_min + frac * (param_max - param_min)
    return round(mapped, 6)

# ============================================================
# NORMALIZATION METHODS
# ============================================================

def normalize_x_values_to_custom_range(xs: list[float], x_min: float, x_max: float) -> list[float]:
    xs = np.array(xs, dtype=np.float64)
    def linear(xs): return (xs - xs.min()) / (xs.max() - xs.min())
    def log(xs):
        xs = np.clip(np.abs(xs), 1e-12, None)
        xs_log = np.log10(xs)
        return (xs_log - xs_log.min()) / (xs_log.max() - xs_log.min())
    def clipped(xs):
        xs = np.clip(xs, -1e6, 1e6)
        return (xs - xs.min()) / (xs.max() - xs.min())
    def tanh_norm(xs):
        t = np.tanh(xs)
        return (t - t.min()) / (t.max() - t.min())
    if NORMALIZATION_MODE == "linear": normalized = linear(xs)
    elif NORMALIZATION_MODE == "log": normalized = log(xs)
    elif NORMALIZATION_MODE == "clipped": normalized = clipped(xs)
    else: normalized = tanh_norm(xs)
    scaled = x_min + normalized * (x_max - x_min)
    return np.round(scaled, 6).tolist()

# ============================================================
# PLOT FUNCTION SYSTEM
# ============================================================

def apply_plot_function(plot_type: int, p1: float, p2: float, x: float) -> float:
    try:
        if plot_type == 0: return p1 * x + p2
        elif plot_type == 1: return p1 * x**2 + p2 * x
        elif plot_type == 2: return p1 * x**3 + p2 * x**2
        elif plot_type == 3: return p1 * math.exp(min(p2 * x, 20))
        elif plot_type == 4: return p1 * math.sin(p2 * x)
        elif plot_type == 5: return p1*x**4 + p2*x**3 + p1*x**2
    except Exception:
        return 0.0
    return 0.0

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes):
    key_base = salt1 + salt2 + char.encode('latin1')
    h_plot = hmac.new(SECRET_KEY, key_base + b"plot_type", hashlib.sha256).digest()
    h_p1 = hmac.new(SECRET_KEY, key_base + b"p1", hashlib.sha256).digest()
    h_p2 = hmac.new(SECRET_KEY, key_base + b"p2", hashlib.sha256).digest()
    h_x = hmac.new(SECRET_KEY, key_base + b"x", hashlib.sha256).digest()
    plot_type = h_plot[0] % 6
    raw_p1 = safe_float_from_bytes(h_p1[:4])
    raw_p2 = safe_float_from_bytes(h_p2[:4])
    p1_min, p1_max = generate_user_param_range(salt1, salt2, b"p1")
    p2_min, p2_max = generate_user_param_range(salt1, salt2, b"p2")
    p1 = normalize_param(raw_p1, p1_min, p1_max)
    p2 = normalize_param(raw_p2, p2_min, p2_max)
    x = safe_float_from_bytes(h_x[:4])
    if DEBUG:
        print(f"[DEBUG] {char!r}: type={plot_type}, p1={p1}({p1_min}-{p1_max}), "
              f"p2={p2}({p2_min}-{p2_max}), x_raw={x}")
    return plot_type, p1, p2, x

# ============================================================
# ENTROPY
# ============================================================

def entropy(data: bytes) -> float:
    counter = Counter(data)
    length = len(data)
    probs = [count / length for count in counter.values()]
    return -sum(p * math.log2(p) for p in probs)

# ============================================================
# BENCHMARKING
# ============================================================

def compare_methods(password: str) -> dict:
    results = {}
    salt = os.urandom(16)
    salt1 = os.urandom(8)
    salt2 = os.urandom(8)
    ph = PasswordHasher(memory_cost=102400, time_cost=2, parallelism=8)

    def measure(fn):
        tracemalloc.start()
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, elapsed, peak / 1024

    # --- SHA256 ---
    digest, t, mem = measure(lambda: hashlib.sha256(password.encode() + salt).digest())
    results["SHA256"] = {"time": t, "entropy": entropy(digest), "output_len": len(digest),
                         "memory": mem, "verified": digest == hashlib.sha256(password.encode() + salt).digest(),
                         "hash": digest.hex()}

    # --- bcrypt ---
    hashed, t, mem = measure(lambda: bcrypt.hashpw(password.encode(), bcrypt.gensalt()))
    results["bcrypt"] = {"time": t, "entropy": entropy(hashed), "output_len": len(hashed),
                         "memory": mem, "verified": bcrypt.checkpw(password.encode(), hashed),
                         "hash": hashed.decode()}

    # --- scrypt ---
    hashed, t, mem = measure(lambda: hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1))
    results["scrypt"] = {"time": t, "entropy": entropy(hashed), "output_len": len(hashed),
                         "memory": mem, "verified": hashed == hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1),
                         "hash": hashed.hex()}

    # --- Argon2 ---
    hash_str, t, mem = measure(lambda: ph.hash(password))
    results["Argon2"] = {"time": t, "entropy": entropy(hash_str.encode()), "output_len": len(hash_str),
                         "memory": mem, "verified": ph.verify(hash_str, password),
                         "hash": hash_str}

    # --- Plot-only ---
    combined_input = password.encode() + salt1 + salt2
    mapped = [map_char_to_function_with_x(chr(b), salt1, salt2) for b in combined_input]
    x_raw = [m[3] for m in mapped]
    x_min, x_max = generate_user_range(salt1, salt2)
    x_norm = normalize_x_values_to_custom_range(x_raw, x_min, x_max)

    def plot_fn():
        values = []
        for i, (plot_type, p1, p2, _) in enumerate(mapped):
            val = apply_plot_function(plot_type, p1, p2, x_norm[i])
            values.append(val)
        binary_data = struct.pack(f'>{len(values)}f', *values)
        return hashlib.sha256(binary_data).digest(), binary_data

    (digest, binary_data), t, mem = measure(plot_fn)
    results["Plot-only"] = {"time": t, "entropy": entropy(digest), "output_len": len(digest),
                            "memory": mem, "verified": digest == hashlib.sha256(binary_data).digest(),
                            "hash": digest.hex(),
                            "binary": binary_data.hex()}

    # --- Plot + Argon2 ---
    hash_str, t, mem = measure(lambda: ph.hash(binary_data.hex()))
    results["Plot+Argon2"] = {"time": t, "entropy": entropy(hash_str.encode()), "output_len": len(hash_str),
                              "memory": mem, "verified": ph.verify(hash_str, binary_data.hex()),
                              "hash": hash_str}

    return results

# ============================================================
# VISUALIZATION WITH PLOTLY
# ============================================================

def interactive_plot(all_results, passwords):
    import plotly.graph_objects as go
    methods = ["SHA256", "bcrypt", "scrypt", "Argon2", "Plot-only", "Plot+Argon2"]
    for metric in ["time", "entropy", "output_len"]:
        fig = go.Figure()
        for m in methods:
            y_vals = [all_results[p][m][metric] for p in passwords]
            hover_texts = []
            for p in passwords:
                if m == "Plot-only":
                    hover_texts.append(f"{m}<br>Password: {p}<br>Binary: {all_results[p][m]['binary']}")
                elif m == "Plot+Argon2":
                    hover_texts.append(f"{m}<br>Password: {p}<br>Argon2: {all_results[p][m]['hash']}")
                else:
                    hover_texts.append(f"{m}<br>Password: {p}<br>Hash: {all_results[p][m]['hash']}")
            fig.add_trace(go.Bar(
                x=passwords,
                y=y_vals,
                name=m,
                text=hover_texts,
                hoverinfo='text'
            ))
        fig.update_layout(title=f"Hashing Method Comparison - {metric}",
                          xaxis_title="Passwords",
                          yaxis_title=metric,
                          barmode='group')
        fig.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    passwords = ["abc", "password123", "12345678", "AbC4%L98\"/?78/IiKkLaDfBn3I0o"]
    all_results = {}
    for pwd in passwords:
        results = compare_methods(pwd)
        all_results[pwd] = results
    interactive_plot(all_results, passwords)
