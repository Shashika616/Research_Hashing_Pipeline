import os
import hmac
import struct
import hashlib
import bcrypt
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
from argon2 import PasswordHasher
from hashlib import sha256
from typing import Tuple
from collections import Counter

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

def safe_float_from_bytes(b: bytes) -> float:
    bits = int.from_bytes(b, byteorder='big')
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0 or exponent == 255:
        exponent = 127
    safe_bits = (sign << 31) | (exponent << 23) | mantissa
    safe_bytes = safe_bits.to_bytes(4, byteorder='big')
    return struct.unpack('>f', safe_bytes)[0]

def generate_user_range(salt1: bytes, salt2: bytes) -> Tuple[float, float]:
    def extract_strong_float(hmac_bytes: bytes) -> float:
        val = safe_float_from_bytes(hmac_bytes[:4])
        abs_val = abs(val) if val != 0 else 1.0
        exponent = math.log10(abs_val)
        scaled = exponent % 3.0
        return 10 ** scaled

    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + b'unpredictable_x_min', sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + b'unpredictable_x_max', sha256).digest()
    min_val = extract_strong_float(hmac_min)
    max_val = extract_strong_float(hmac_max)
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    if abs(max_val - min_val) < 1.0:
        separation_hmac = hmac.new(SECRET_KEY, salt1 + salt2 + b'separation', sha256).digest()
        separation = round(safe_float_from_bytes(separation_hmac[:4]) % 10.0, 6)
        max_val = round(min_val + separation + 1.5, 6)
    return round(min_val, 6), round(max_val, 6)

def normalize_x_values_to_custom_range(xs: list[float], x_min: float, x_max: float) -> list[float]:
    """
    Supports multiple normalization strategies. Switch between them by commenting/uncommenting.
    """

    xs = np.array(xs, dtype=np.float64)

    ### --- Option 1: Original Linear Normalization ---
    # Purpose: simple min-max normalization
    # Strengths: fast, straightforward
    # Weakness: fails with extreme outliers
    def linear_normalization(xs):
        x_raw_min = np.min(xs)
        x_raw_max = np.max(xs)
        if abs(x_raw_max - x_raw_min) < 1e-6:
            x_raw_max = x_raw_min + 1.0
        normalized = (xs - x_raw_min) / (x_raw_max - x_raw_min)
        return x_min + normalized * (x_max - x_min)

    ### --- Option 2: Logarithmic Normalization ---
    # Purpose: compresses wide-magnitude values
    # Strengths: handles extreme differences well
    # Weakness: loses sign (uses abs), needs log-safe values
    def log_normalization(xs):
        safe_vals = np.clip(np.abs(xs), 1e-30, 1e+30)
        log_vals = np.log10(safe_vals)
        log_min = np.min(log_vals)
        log_max = np.max(log_vals)
        if abs(log_max - log_min) < 1e-6:
            log_max = log_min + 1.0
        normalized = (log_vals - log_min) / (log_max - log_min)
        return x_min + normalized * (x_max - x_min)

    ### --- Option 3: Clipping-Based Normalization ---
    # Purpose: remove impact of extreme outliers
    # Strengths: great for heavily skewed values
    # Weakness: clipping may lose useful high-magnitude info
    def clipped_normalization(xs):
        lower, upper = np.percentile(xs, [1, 99])
        clipped = np.clip(xs, lower, upper)
        normed = (clipped - lower) / (upper - lower)
        return x_min + normed * (x_max - x_min)

    ### --- Option 4: Tanh-Based Normalization ---
    # Purpose: squashes values into (-1, 1)
    # Strengths: strong outlier control, keeps sign
    # Weakness: output distribution may concentrate around center
    def tanh_normalization(xs):
        scale = 1e10  # Can tune this as needed
        squashed = np.tanh(xs / scale)
        tanh_min = np.min(squashed)
        tanh_max = np.max(squashed)
        if abs(tanh_max - tanh_min) < 1e-6:
            tanh_max = tanh_min + 1.0
        normed = (squashed - tanh_min) / (tanh_max - tanh_min)
        return x_min + normed * (x_max - x_min)

    # === Select which one to use ===
    # normalized = linear_normalization(xs)       # <- Default (Option 1)
    normalized = log_normalization(xs)        # <- Uncomment for Option 2
    # normalized = clipped_normalization(xs)    # <- Uncomment for Option 3
    # normalized = tanh_normalization(xs)       # <- Uncomment for Option 4

    return np.round(normalized, 6).tolist()

def apply_plot_function(plot_type: int, p1: float, p2: float, x: float) -> float:
    try:
        if plot_type == 0:
            return p1 * x + p2
        elif plot_type == 1:
            return p1 * x**2 + p2 * x
        elif plot_type == 2:
            return p1 * x**3 + p2 * x**2
        elif plot_type == 3:
            return p1 * (2.71828 ** (p2 * x))
        elif plot_type == 4:
            from math import sin
            return p1 * sin(p2 * x)
        elif plot_type == 5:
            return p1 * x**4 + p2 * x**3 + p1 * x**2
        else:
            return 0.0
    except:
        return 0.0

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes) -> Tuple[int, float, float, float]:
    char_bytes = char.encode('utf-8')
    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'type', sha256).digest()
    plot_type = hmac_type[0] % 6
    hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p1', sha256).digest()
    p1 = safe_float_from_bytes(hmac_p1[:4])
    hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p2', sha256).digest()
    p2 = safe_float_from_bytes(hmac_p2[:4])
    hmac_x = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'x', sha256).digest()
    x = safe_float_from_bytes(hmac_x[:4])
    return plot_type, p1, p2, x

def entropy(data: bytes) -> float:
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

def compare_methods(password: str):
    salt = os.urandom(16)
    salt1 = os.urandom(8)
    salt2 = os.urandom(8)
    results = {}

    # SHA256
    tracemalloc.start()
    start = time.time()
    sha = hashlib.sha256(password.encode() + salt).digest()
    sha_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    sha_entropy = entropy(sha)
    results['SHA256'] = {
        'time': sha_time, 'entropy': sha_entropy,
        'output_len': len(sha), 'memory': f"{peak / 1024:.3f} KB",
        'verified': hashlib.sha256(password.encode() + salt).digest() == sha,
        'hash': sha.hex(), 'salt': salt.hex()
    }

    # bcrypt
    tracemalloc.start()
    start = time.time()
    bcrypt_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    bcrypt_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    bcrypt_entropy = entropy(bcrypt_hash)
    results['bcrypt'] = {
        'time': bcrypt_time, 'entropy': bcrypt_entropy,
        'output_len': len(bcrypt_hash), 'memory': f"{peak / 1024:.3f} KB",
        'verified': bcrypt.checkpw(password.encode(), bcrypt_hash),
        'hash': bcrypt_hash.decode(), 'salt': bcrypt_hash[:29].decode()
    }

    # scrypt
    scrypt_n, scrypt_r, scrypt_p = 2**14, 8, 1
    tracemalloc.start()
    start = time.time()
    scrypt_hash = hashlib.scrypt(password.encode(), salt=salt, n=scrypt_n, r=scrypt_r, p=scrypt_p)
    scrypt_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    scrypt_entropy = entropy(scrypt_hash)
    results['scrypt'] = {
        'time': scrypt_time, 'entropy': scrypt_entropy,
        'output_len': len(scrypt_hash), 'memory': f"{peak / 1024:.3f} KB",
        'verified': hashlib.scrypt(password.encode(), salt=salt, n=scrypt_n, r=scrypt_r, p=scrypt_p) == scrypt_hash,
        'hash': scrypt_hash.hex(), 'salt': salt.hex()
    }

    # Argon2
    hasher = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=8)
    tracemalloc.start()
    start = time.time()
    argon_hash = hasher.hash(password)
    argon_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    argon_entropy = entropy(argon_hash.encode())
    try:
        verified = hasher.verify(argon_hash, password)
    except:
        verified = False
    results['Argon2'] = {
        'time': argon_time, 'entropy': argon_entropy,
        'output_len': len(argon_hash.encode()), 'memory': f"{peak / 1024:.3f} KB",
        'verified': verified, 'hash': argon_hash, 'salt': "Embedded"
    }

    # Plot-only
    combined_input = password.encode() + salt1 + salt2
    x_raw_values = []
    for b in combined_input:
        char = chr(b)
        _, _, _, x = map_char_to_function_with_x(char, salt1, salt2)
        x_raw_values.append(x)

    user_x_min, user_x_max = generate_user_range(salt1, salt2)
    x_values = normalize_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max)

    values = []
    for i, b in enumerate(combined_input):
        char = chr(b)
        plot_type, p1, p2, _ = map_char_to_function_with_x(char, salt1, salt2)
        result = apply_plot_function(plot_type, p1, p2, x_values[i])
        values.append(result)

    binary_data = struct.pack(f'{len(values)}f', *values)
    tracemalloc.start()
    start = time.time()
    plot_hash = hashlib.sha256(binary_data).digest()
    plot_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results['Plot-only'] = {
        'time': plot_time, 'entropy': entropy(binary_data),
        'output_len': len(plot_hash), 'memory': f"{peak / 1024:.3f} KB",
        'verified': hashlib.sha256(binary_data).digest() == plot_hash,
        'hash': plot_hash.hex(), 'salt': f"{salt1.hex()} + {salt2.hex()}"
    }

    # Plot+Argon2
    tracemalloc.start()
    start = time.time()
    plot_argon_hash = hasher.hash(binary_data)
    plot_argon_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    try:
        verified = hasher.verify(plot_argon_hash, binary_data)
    except:
        verified = False
    results['Plot+Argon2'] = {
        'time': plot_argon_time, 'entropy': entropy(plot_argon_hash.encode()),
        'output_len': len(plot_argon_hash.encode()), 'memory': f"{peak / 1024:.3f} KB",
        'verified': verified, 'hash': plot_argon_hash, 'salt': f"{salt1.hex()} + {salt2.hex()}"
    }

    return results, x_raw_values, x_values, user_x_min, user_x_max

def print_results(results, password):
    print(f"\n=== Results for password: '{password}' ===")
    print(f"{'Method':<12} | {'Time (s)':<10} | {'Entropy':<10} | {'Output Len':<10} | {'Memory Usage':<25} | {'Verified':<8} | {'Sample Hash (truncated)'}")
    print("-"*110)
    for method, data in results.items():
        truncated_hash = (data['hash'][:40] + '...') if len(data['hash']) > 40 else data['hash']
        print(f"{method:<12} | {data['time']:<10.6f} | {data['entropy']:<10.4f} | {data['output_len']:<10} | {data['memory']:<25} | {str(data['verified']):<8} | {truncated_hash}")
        print(f"{'':<12}   Full Hash: {data['hash']}")
        print(f"{'':<12}   Salt     : {data['salt']}\n")

def plot_x_distributions(password: str, x_raw_values: list[float], x_normalized_values: list[float], x_min: float, x_max: float):
    x_indices = list(range(len(x_raw_values)))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_indices, x_raw_values, marker='o', linestyle='-', color='tab:red')
    plt.title(f"Raw x-values: '{password}'")
    plt.xlabel("Index")
    plt.ylabel("Raw x")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_indices, x_normalized_values, marker='o', linestyle='-', color='tab:blue')
    plt.axhline(y=x_min, color='green', linestyle='--', label=f"x_min = {x_min}")
    plt.axhline(y=x_max, color='orange', linestyle='--', label=f"x_max = {x_max}")
    plt.title(f"Normalized x-values: '{password}'")
    plt.xlabel("Index")
    plt.ylabel("Normalized x")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    passwords = ["abc", "password123", "12345678", "AbC4%L98/?78/IiKkLaDfBn3I0o"]
    methods = ["SHA256", "bcrypt", "scrypt", "Argon2", "Plot-only", "Plot+Argon2"]
    all_times = {m: [] for m in methods}
    all_entropies = {m: [] for m in methods}
    all_output_lens = {m: [] for m in methods}

    for pwd in passwords:
        results, x_raw, x_norm, xmin, xmax = compare_methods(pwd)
        print_results(results, pwd)
        plot_x_distributions(pwd, x_raw, x_norm, xmin, xmax)

        for method in methods:
            all_times[method].append(results[method]['time'])
            all_entropies[method].append(results[method]['entropy'])
            all_output_lens[method].append(results[method]['output_len'])
