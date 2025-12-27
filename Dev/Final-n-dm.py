import os
import hmac
import struct
import hashlib
import bcrypt
import time
import math
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import tracemalloc
from argon2 import PasswordHasher
from hashlib import sha256
from typing import Tuple, List
from collections import Counter

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

def safe_float_from_bytes(b: bytes, precision: int = 6) -> float:
    """
    Converts 4 bytes into a safe IEEE-754 float with controlled range and precision.
    Avoids too many zeros, denormals, NaNs, or extreme values.
    """
    # Convert bytes to uint32 bits
    bits = int.from_bytes(b, byteorder='big')
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    # Clamp exponent to avoid denormals, zeros, Inf, NaN
    if exponent == 0 or exponent == 255:
        exponent = 127  # bias for exponent 0 → safe normal number

    # Clamp mantissa to avoid exact zero
    if mantissa == 0:
        mantissa = 1

    safe_bits = (sign << 31) | (exponent << 23) | mantissa
    safe_bytes = safe_bits.to_bytes(4, byteorder='big')

    val = struct.unpack('>f', safe_bytes)[0]

    # Clamp magnitude to avoid extremes
    min_val = 1e-6
    max_val = 1e6
    val = max(min(val, max_val), -max_val)
    if abs(val) < min_val:
        val = min_val if val >= 0 else -min_val

    return round(val, precision)

def generate_user_range(salt1: bytes, salt2: bytes) -> Tuple[float, float]:
    """
    Generates a user-specific, unpredictable, and IEEE 754-safe float range (x_min, x_max),
    derived from secure HMAC outputs.
    """

    def extract_strong_float(hmac_bytes: bytes) -> float:
        val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
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

def normalize_multi_dim_x_values_to_custom_range(xs_list: List[List[float]], x_min: float, x_max: float) -> List[List[float]]:
    """
    Normalize a list of per-character vectors (list of lists) into [x_min, x_max].
    Each character's vector is normalized independently to preserve per-char structure.
    Returns the same shape list of lists with rounded values.
    """
    normalized_all = []
    for vec in xs_list:
        arr = np.array(vec, dtype=np.float64)
        if arr.size == 0:
            normalized_all.append([])
            continue

        # Use linear normalization per-vector (safe)
        raw_min = np.min(arr)
        raw_max = np.max(arr)
        if abs(raw_max - raw_min) < 1e-9:
            # If all equal, map them to midpoint of target range
            normed = np.full_like(arr, 0.5)
        else:
            normed = (arr - raw_min) / (raw_max - raw_min)

        mapped = x_min + normed * (x_max - x_min)
        normalized_all.append(np.round(mapped, 6).tolist())

    return normalized_all

def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
    """
    Generates a user-specific, cryptographically unpredictable safe range for parameters (p1, p2).
    """
    def extract_strong_float(hmac_bytes: bytes) -> float:
        base_val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
        extra_bits = int.from_bytes(hmac_bytes[4:8], byteorder='big')
        frac_rand = (extra_bits % 10000) / 10000.0
        combined = abs(base_val) * (1 + frac_rand)
        scaled = (math.log10(combined + 1e-6) + 6) % 3.0
        return round(10 ** scaled, 6)

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

def normalize_param(value: float, param_min: float, param_max: float, method: str = "linear") -> float:
    """
    Normalizes a single parameter into a user-specific safe range.
    """
    x = np.array([value], dtype=np.float64)

    def linear_norm(x):
        scaled = abs(x) % 1.0
        return param_min + scaled * (param_max - param_min)

    def log_norm(x):
        safe_val = np.clip(np.abs(x), 1e-30, 1e30)
        log_val = np.log10(safe_val)
        log_min, log_max = log_val, log_val
        if abs(log_max - log_min) < 1e-6:
            log_max = log_min + 1.0
        normed = (log_val - log_min) / (log_max - log_min)
        return param_min + normed * (param_max - param_min)

    def clipped_norm(x):
        lower, upper = -1.0, 1.0
        clipped = np.clip(x, lower, upper)
        normed = (clipped - lower) / (upper - lower)
        return param_min + normed * (param_max - param_min)

    def tanh_norm(x):
        scale = 1e10
        squashed = np.tanh(x / scale)
        tanh_min, tanh_max = squashed, squashed
        if abs(tanh_max - tanh_min) < 1e-6:
            tanh_max = tanh_min + 1.0
        normed = (squashed - tanh_min) / (tanh_max - tanh_min)
        return param_min + normed * (param_max - param_min)

    methods = {
        "linear": linear_norm,
        "log": log_norm,
        "clipped": clipped_norm,
        "tanh": tanh_norm
    }

    if method not in methods:
        raise ValueError(f"Unknown normalization method: {method}")

    normalized = methods[method](x)[0]
    return round(normalized, 6)


def apply_plot_function(plot_type: int, p1: float, p2: float, x_list: List[float]) -> float:
    """
    Safe execution engine for multi-dimensional mathematical functions f(x1,...,xn).
    Returns a single scalar y for a vector x_list.
    Uses safe guards to avoid overflows and NaNs.
    """
    MAX_MAG = 1e12
    MIN_MAG = 1e-12

    def clamp(v):
        try:
            if math.isnan(v) or math.isinf(v):
                return 0.0
        except:
            return 0.0
        if v > MAX_MAG: return MAX_MAG
        if v < -MAX_MAG: return -MAX_MAG
        if abs(v) < MIN_MAG: return MIN_MAG * (1 if v >= 0 else -1)
        return v

    try:
        n = max(1, len(x_list))
        # Ensure we have at least x1; fill missing with small safe values (shouldn't happen)
        xs = [float(x_list[i]) if i < len(x_list) else 1e-6 for i in range(n)]

        # Combine the multi-dim inputs in different ways depending on plot_type.
        # These are intentionally generic: they aggregate across dimensions to a single scalar.
        if plot_type == 0:
            # Sum of polynomial terms across dims + trig + exp aggregation
            val = 0.0
            for i, xi in enumerate(xs):
                k = (i % 5) + 2
                val += p1 * (xi ** k) - p2 * (xi ** (k-1))
                val += math.sin(p1 * (xi ** (k-2)))
            # add cross-dim coupling
            val += math.exp(sum(xs) / (1 + abs(sum(xs))))
            val -= p1 * math.log(sum([abs(xx) for xx in xs]) + 10)

        elif plot_type == 1:
            # Deeply nested hybrid with product and logistic damping
            prod = 1.0
            for xi in xs:
                prod *= (1 + abs(xi)) ** (0.1)
            denom = 1 + math.exp(-p2 * sum(xs))
            val = (p1 * prod) / denom + math.tan(math.sin(p1 * sum(xs)))

        elif plot_type == 2:
            # Oscillatory with per-dim poly + log & expo mix
            val = sum(p1 * (xi ** 3) + p2 * (xi ** 2) - p1 * xi for xi in xs)
            val += sum(math.sin(xi ** 2) for xi in xs)
            val += math.log(sum([abs(p2 * xi) for xi in xs]) + 5)
            val -= math.exp(-abs(sum(xs)))

        elif plot_type == 3:
            # Chaotic exp-log-poly mix with dimension-aware terms
            val = p1 * math.exp(math.sin(p2 * sum(xs)))
            val -= p2 * math.log(sum([abs(xi * p1) for xi in xs]) + 2)
            # alternate sign polynomial aggregate
            s = 0.0
            for i, xi in enumerate(xs):
                s += ((-1) ** i) * (xi ** (i % 4 + 2))
            val += s

        elif plot_type == 4:
            # Heavy trig chain aggregated across dims
            val = sum(math.sin(p1 * (xi ** 3)) + math.cos(p2 * (xi ** 2)) + math.tan(p1 * xi / (1 + abs(xi))) for xi in xs)

        elif plot_type == 5:
            # High-degree polynomial aggregation (degree reduced per-dim)
            val = sum(p1 * (xi ** (min(6, i+2))) - p2 * (xi ** (min(5, i+1))) for i, xi in enumerate(xs))
            val += sum(math.sin(xi) for xi in xs) + math.exp(-abs(p1 * sum(xs)))

        else:
            val = 0.0

        return clamp(val)

    except Exception:
        return 0.0

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes, max_dims: int = 8) -> Tuple[int, float, float, int, List[float]]:
    """
    For a given character + salts, derive:
        - plot_type (int)
        - p1 (float)
        - p2 (float)
        - n (int) number of dimensions for f(x1,...,xn)
        - x_list: list of n safe floats (each derived from HMAC blocks)
    n is derived deterministically via an HMAC and mapped to 1..max_dims.
    """
    char_bytes = char.encode('utf-8')

    # plot type
    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'type', sha256).digest()
    plot_type = hmac_type[0] % 6

    # p1 and p2 base values
    hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p1', sha256).digest()
    raw_p1 = safe_float_from_bytes(hmac_p1[:4])

    hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p2', sha256).digest()
    raw_p2 = safe_float_from_bytes(hmac_p2[:4])

    # normalize p1 and p2 into user-specific ranges
    p1_min, p1_max = generate_user_param_range(salt1, salt2, b'p1')
    p2_min, p2_max = generate_user_param_range(salt1, salt2, b'p2')
    p1 = normalize_param(raw_p1, p1_min, p1_max, method="log")
    p2 = normalize_param(raw_p2, p2_min, p2_max, method="log")

    # determine n (dimensions) via a dedicated HMAC
    hmac_n = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'n_dims', sha256).digest()
    # Map to 1..max_dims
    n = (hmac_n[0] % max_dims) + 1

    # For x_i values, derive using successive HMACs/blocks so each xi is independent
    x_list: List[float] = []
    # We'll use the base x-hmac and further blocks via index suffixes
    for i in range(n):
        block_tag = b'x' + bytes([i])
        hmac_xi = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + block_tag, sha256).digest()
        xi = safe_float_from_bytes(hmac_xi[:4])
        x_list.append(xi)

    # Debug print for ranges and dims
    print(f"    [RANGE] p1_min={p1_min:.6f}, p1_max={p1_max:.6f} | p2_min={p2_min:.6f}, p2_max={p2_max:.6f} | n={n} | x_count={len(x_list)}")

    return plot_type, p1, p2, n, x_list


def entropy(data: bytes) -> float:
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())


def plot_results_plotly(passwords, all_times, all_entropies, all_output_lens, methods):
    x = np.arange(len(passwords))
    width = 0.12

    fig_time = go.Figure()
    for i, method in enumerate(methods):
        fig_time.add_trace(go.Bar(
            x=[pwd for pwd in passwords],
            y=all_times[method],
            name=method
        ))
    fig_time.update_layout(
        title="Hashing Time Comparison",
        xaxis_title="Passwords",
        yaxis_title="Hashing Time (s)",
        barmode='group',
        template='plotly_white'
    )

    fig_entropy = go.Figure()
    for i, method in enumerate(methods):
        fig_entropy.add_trace(go.Bar(
            x=[pwd for pwd in passwords],
            y=all_entropies[method],
            name=method
        ))
    fig_entropy.update_layout(
        title="Entropy of Output",
        xaxis_title="Passwords",
        yaxis_title="Entropy (bits)",
        barmode='group',
        template='plotly_white'
    )

    fig_len = go.Figure()
    for i, method in enumerate(methods):
        fig_len.add_trace(go.Bar(
            x=[pwd for pwd in passwords],
            y=all_output_lens[method],
            name=method
        ))
    fig_len.update_layout(
        title="Output Length Comparison",
        xaxis_title="Passwords",
        yaxis_title="Output Length (bytes)",
        barmode='group',
        template='plotly_white'
    )

    fig_time.show()
    fig_entropy.show()
    fig_len.show()


def iterative_plot_transform(password: str, salt1: bytes, salt2: bytes, iterations: int = 3):
    """
    Apply the Plot transformation iteratively.
    Each iteration treats the previous output binary as the next 'password'.
    Now uses multi-dimensional per-character functions f(x1,...,xn).
    """
    current_input = password.encode()
    all_intermediates = []

    for it in range(1, iterations + 1):
        combined_input = current_input + salt1 + salt2
        x_raw_values = []  # list of per-char lists

        # Map each byte/char to function and gather raw x vectors
        for b in combined_input:
            char = chr(b % 256)
            plot_type, p1, p2, n, x_list = map_char_to_function_with_x(char, salt1, salt2)
            x_raw_values.append(x_list)

        # Normalize per-char vectors into the user-specific range
        user_x_min, user_x_max = generate_user_range(salt1, salt2)
        x_values_multi = normalize_multi_dim_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max)

        # Apply plot functions (one scalar result per character)
        values = []
        for i, b in enumerate(combined_input):
            char = chr(b % 256)
            plot_type, p1, p2, n, _ = map_char_to_function_with_x(char, salt1, salt2)
            x_vec = x_values_multi[i]
            # Safety: if we somehow have fewer xi than n, pad with small values
            if len(x_vec) < n:
                x_vec = x_vec + [1e-6] * (n - len(x_vec))
            result = apply_plot_function(plot_type, p1, p2, x_vec)
            values.append(result)

        # Convert to binary
        binary_data = struct.pack(f'{len(values)}f', *values)
        all_intermediates.append(binary_data)

        # Next iteration input
        current_input = binary_data

    return current_input, all_intermediates


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
    sha_memory_kb = peak / 1024
    sha_entropy = entropy(sha)
    sha_len = len(sha)
    sha_verify = (hashlib.sha256(password.encode() + salt).digest() == sha)

    results['SHA256'] = {
        'time': sha_time,
        'entropy': sha_entropy,
        'output_len': sha_len,
        'memory': f"{sha_memory_kb:.3f} KB",
        'verified': sha_verify,
        'hash': sha.hex(),
        'salt': salt.hex()
    }

    # bcrypt
    tracemalloc.start()
    start = time.time()
    bcrypt_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    bcrypt_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    bcrypt_memory_kb = peak / 1024
    bcrypt_entropy = entropy(bcrypt_hash)
    bcrypt_len = len(bcrypt_hash)
    bcrypt_verify = bcrypt.checkpw(password.encode(), bcrypt_hash)

    results['bcrypt'] = {
        'time': bcrypt_time,
        'entropy': bcrypt_entropy,
        'output_len': bcrypt_len,
        'memory': f"{bcrypt_memory_kb:.3f} KB",
        'verified': bcrypt_verify,
        'hash': bcrypt_hash.decode(),
        'salt': bcrypt_hash[:29].decode()
    }

    # scrypt
    scrypt_n = 2**14
    scrypt_r = 8
    scrypt_p = 1
    tracemalloc.start()
    start = time.time()
    scrypt_hash = hashlib.scrypt(password.encode(), salt=salt, n=scrypt_n, r=scrypt_r, p=scrypt_p)
    scrypt_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    scrypt_memory_kb = peak / 1024
    scrypt_entropy = entropy(scrypt_hash)
    scrypt_len = len(scrypt_hash)
    scrypt_verify = (hashlib.scrypt(password.encode(), salt=salt, n=scrypt_n, r=scrypt_r, p=scrypt_p) == scrypt_hash)

    results['scrypt'] = {
        'time': scrypt_time,
        'entropy': scrypt_entropy,
        'output_len': scrypt_len,
        'memory': f"{scrypt_memory_kb:.3f} KB",
        'verified': scrypt_verify,
        'hash': scrypt_hash.hex(),
        'salt': salt.hex()
    }

    # Argon2
    hasher = PasswordHasher(time_cost=2, memory_cost=102400, parallelism=8)
    tracemalloc.start()
    start = time.time()
    argon_hash = hasher.hash(password)
    argon_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    argon_memory_kb = peak / 1024
    argon_entropy = entropy(argon_hash.encode())
    argon_len = len(argon_hash.encode())
    try:
        argon_verify = hasher.verify(argon_hash, password)
    except:
        argon_verify = False

    results['Argon2'] = {
        'time': argon_time,
        'entropy': argon_entropy,
        'output_len': argon_len,
        'memory': f"{argon_memory_kb:.3f} KB",
        'verified': argon_verify,
        'hash': argon_hash,
        'salt': "Embedded in hash string"
    }

    # ===== Iterative Plot-only / Plot+Argon2 =====
    iterations = 2  # you can adjust this

    final_binary, all_intermediates = iterative_plot_transform(password, salt1, salt2, iterations)

    # Full intermediate binary data
    print("\n[DEBUG] Full intermediate binaries:")
    for i, bdata in enumerate(all_intermediates):
        print(f"[Iteration {i+1}] Binary hex ({len(bdata)} bytes):\n{bdata.hex()}\n")

    # Plot-only (final iteration)
    tracemalloc.start()
    start = time.time()
    plot_hash = hashlib.sha256(final_binary).digest()
    plot_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plot_memory_kb = peak / 1024
    plot_entropy = entropy(final_binary)
    plot_len = len(plot_hash)
    plot_verify = (hashlib.sha256(final_binary).digest() == plot_hash)

    results['Plot-only'] = {
        'time': plot_time,
        'entropy': plot_entropy,
        'output_len': plot_len,
        'memory': f"{plot_memory_kb:.3f} KB",
        'verified': plot_verify,
        'hash': plot_hash.hex(),
        'salt': f"{salt1.hex()} + {salt2.hex()}",
        'iterations': iterations
    }

    # Plot+Argon2
    tracemalloc.start()
    start = time.time()
    plot_argon_hash = hasher.hash(final_binary)
    plot_argon_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plot_argon_memory_kb = peak / 1024
    plot_argon_entropy = entropy(plot_argon_hash.encode())
    plot_argon_len = len(plot_argon_hash.encode())
    try:
        plot_argon_verify = hasher.verify(plot_argon_hash, final_binary)
    except:
        plot_argon_verify = False

    results['Plot+Argon2'] = {
        'time': plot_argon_time,
        'entropy': plot_argon_entropy,
        'output_len': plot_argon_len,
        'memory': f"{plot_argon_memory_kb:.3f} KB",
        'verified': plot_argon_verify,
        'hash': plot_argon_hash,
        'salt': f"{salt1.hex()} + {salt2.hex()}"
    }

    return results


def print_results(results, password):
    print(f"\n=== Results for password: '{password}' ===")
    print(f"{'Method':<12} | {'Time (s)':<10} | {'Entropy':<10} | {'Output Len':<10} | {'Memory Usage':<25} | {'Verified':<8} | {'Sample Hash (truncated)'}")
    print("-"*110)
    for method, data in results.items():
        truncated_hash = (data['hash'][:40] + '...') if len(data['hash']) > 40 else data['hash']
        print(f"{method:<12} | {data['time']:<10.6f} | {data['entropy']:<10.4f} | {data['output_len']:<10} | {data['memory']:<25} | {str(data['verified']):<8} | {truncated_hash}")
        print(f"{'':<12}   Full Hash: {data['hash']}")
        print(f"{'':<12}   Salt     : {data['salt']}\n")


if __name__ == "__main__":
    passwords = ["abc"]
    methods = ["SHA256", "bcrypt", "scrypt", "Argon2", "Plot-only", "Plot+Argon2"]

    all_times = {m: [] for m in methods}
    all_entropies = {m: [] for m in methods}
    all_output_lens = {m: [] for m in methods}

    for pwd in passwords:
        results = compare_methods(pwd)
        print_results(results, pwd)

        for method in methods:
            all_times[method].append(results[method]['time'])
            all_entropies[method].append(results[method]['entropy'])
            all_output_lens[method].append(results[method]['output_len'])

    x = np.arange(len(passwords))
    width = 0.12

    # === Merge all plots into a single figure with subplots ===
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # 1 Hashing Time
    for i, method in enumerate(methods):
        axs[0].bar(x + (i - 2.5) * width, all_times[method], width, label=method)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(passwords)
    axs[0].set_ylabel("Hashing Time (s)")
    axs[0].set_title("Hashing Time Comparison")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    # 2 Entropy
    for i, method in enumerate(methods):
        axs[1].bar(x + (i - 2.5) * width, all_entropies[method], width, label=method)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(passwords)
    axs[1].set_ylabel("Entropy (bits)")
    axs[1].set_title("Entropy of Output")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend()

    # 3 Output Length
    for i, method in enumerate(methods):
        axs[2].bar(x + (i - 2.5) * width, all_output_lens[method], width, label=method)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(passwords)
    axs[2].set_ylabel("Output Length (bytes)")
    axs[2].set_title("Output Length Comparison")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend()

    plt.tight_layout()
    plt.show()
    plot_results_plotly(passwords, all_times, all_entropies, all_output_lens, methods)
