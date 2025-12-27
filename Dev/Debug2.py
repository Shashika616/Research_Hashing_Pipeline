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

# def safe_float_from_bytes(b: bytes) -> float:
#     # Convert bytes to uint32
#     bits = int.from_bytes(b, byteorder='big')
#     sign = (bits >> 31) & 0x1
#     exponent = (bits >> 23) & 0xFF
#     mantissa = bits & 0x7FFFFF
#     # Clamp exponent to avoid denormals, zeros, Inf, NaN
#     if exponent == 0 or exponent == 255:
#         exponent = 127  # bias for exponent 0 -> normal number near 1.0
#     safe_bits = (sign << 31) | (exponent << 23) | mantissa
#     safe_bytes = safe_bits.to_bytes(4, byteorder='big')
#     val = struct.unpack('>f', safe_bytes)[0]
#     return val

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
    derived from secure HMAC outputs. Ensures the range is not too narrow or too large
    for safe mathematical operations in plot-based password hashing.
    """

    def extract_strong_float(hmac_bytes: bytes) -> float:
        """
        Converts 4 bytes of HMAC output into a safe float value.
        Scales it logarithmically to keep within a safe range (≈ 1.0 to 1000.0),
        avoiding float overflows, underflows, NaNs, or denormals.
        """
        val = safe_float_from_bytes(hmac_bytes[:4], precision=6)  # Convert raw bytes to float
        abs_val = abs(val) if val != 0 else 1.0      # Avoid log10(0)
        exponent = math.log10(abs_val)               # Get magnitude
        scaled = exponent % 3.0                      # Clamp to [0, 3)
        return 10 ** scaled                          # Result in [1.0, 1000.0)

    # Step 1: Generate HMAC digests using salts and context strings to get unique values
    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + b'unpredictable_x_min', sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + b'unpredictable_x_max', sha256).digest()

    # Step 2: Convert those digests to safe floats using the helper
    min_val = extract_strong_float(hmac_min)
    max_val = extract_strong_float(hmac_max)

    # Step 3: Swap if max < min to maintain correct order
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    # Step 4: Ensure range has a minimum gap; too small a range causes bad normalization
    if abs(max_val - min_val) < 1.0:
        # Add additional gap derived from another HMAC to keep it user-specific
        separation_hmac = hmac.new(SECRET_KEY, salt1 + salt2 + b'separation', sha256).digest()
        separation = round(safe_float_from_bytes(separation_hmac[:4]) % 10.0, 6)  # Add up to 10 units
        max_val = round(min_val + separation + 1.5, 6)  # Final adjusted max value

    return round(min_val, 6), round(max_val, 6)




# def normalize_x_values_to_custom_range(xs: list[float], x_min: float, x_max: float) -> list[float]:
#     x_raw_min = min(xs)
#     x_raw_max = max(xs)
#     if abs(x_raw_max - x_raw_min) < 1e-6:
#         x_raw_max = x_raw_min + 1.0  # Avoid divide-by-zero

#     normalized_xs = []
#     for x in xs:
#         normalized = (x - x_raw_min) / (x_raw_max - x_raw_min)
#         mapped_x = x_min + normalized * (x_max - x_min)
#         normalized_xs.append(round(mapped_x, 6))

#     return normalized_xs

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
        scale = 1e10  # Can tune this
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

def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
    """
    Generates a user-specific, cryptographically unpredictable safe range for parameters (p1, p2).
    The randomness is derived from HMACs of salts + label, then mapped into a safe [a, b] range
    without clustering near 1.0.
    """

    def extract_strong_float(hmac_bytes: bytes) -> float:
        # Convert bytes into a safe float
        base_val = safe_float_from_bytes(hmac_bytes[:4], precision=6)

        # Derive a secondary randomness factor from additional bits
        extra_bits = int.from_bytes(hmac_bytes[4:8], byteorder='big')
        frac_rand = (extra_bits % 10000) / 10000.0  # → [0, 1)

        # Combine both into a pseudo-random safe float, then scale logarithmically
        combined = abs(base_val) * (1 + frac_rand)

        # Log scaling keeps it within meaningful numeric bounds but avoids clustering
        scaled = (math.log10(combined + 1e-6) + 6) % 3.0  # result ∈ [0, 3)
        return round(10 ** scaled, 6)  # → range roughly [1, 1000)

    # --- Derive cryptographically strong seeds for both bounds
    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_min', sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_max', sha256).digest()

    min_val = extract_strong_float(hmac_min)
    max_val = extract_strong_float(hmac_max)

    # --- Ensure correct ordering and minimum separation
    if max_val < min_val:
        min_val, max_val = max_val, min_val

    # Enforce a nontrivial spread
    if abs(max_val - min_val) < 5.0:
        sep_hmac = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_sep', sha256).digest()
        separation = (int.from_bytes(sep_hmac[:2], 'big') % 50) / 10.0  # → up to +5.0 spread
        max_val = round(min_val + 5.0 + separation, 6)

    return round(min_val, 6), round(max_val, 6)

def normalize_param(value: float, param_min: float, param_max: float) -> float:
    """
    Normalizes a single parameter into a user-specific safe range.
    """
    # Safe default: linear scaling of |value|
    abs_val = abs(value)
    # Bring into [0,1]
    scaled = (abs_val % 1.0)
    # Map into [param_min, param_max]
    mapped = param_min + scaled * (param_max - param_min)
    return round(mapped, 6)


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
            from math import log
            # return p1 * log(p2 * x) if p2 * x > 0 else 0.0
            return p1 * x**4 + p2 * x**3 + p1* x**2
        else:
            return 0.0
    except:
        return 0.0

# def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes) -> Tuple[int, float, float, float]:
#     char_bytes = char.encode('utf-8')

#     hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'type', sha256).digest()
#     plot_type = hmac_type[0] % 6

#     hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p1', sha256).digest()
#     p1 = safe_float_from_bytes(hmac_p1[:4])

#     hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p2', sha256).digest()
#     p2 = safe_float_from_bytes(hmac_p2[:4])

#     hmac_x = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'x', sha256).digest()
#     x = safe_float_from_bytes(hmac_x[:4])

#     return plot_type, p1, p2, x

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes) -> Tuple[int, float, float, float]:
    char_bytes = char.encode('utf-8')

    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'type', sha256).digest()
    plot_type = hmac_type[0] % 6

    hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p1', sha256).digest()
    raw_p1 = safe_float_from_bytes(hmac_p1[:4])

    hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p2', sha256).digest()
    raw_p2 = safe_float_from_bytes(hmac_p2[:4])

    # === New: normalize p1 and p2 into user-specific ranges ===
    p1_min, p1_max = generate_user_param_range(salt1, salt2, b'p1')
    p2_min, p2_max = generate_user_param_range(salt1, salt2, b'p2')

    p1 = normalize_param(raw_p1, p1_min, p1_max)
    p2 = normalize_param(raw_p2, p2_min, p2_max)

    hmac_x = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'x', sha256).digest()
    x = safe_float_from_bytes(hmac_x[:4])

     # 🔹 Debug print for ranges
    print(f"    [RANGE] p1_min={p1_min:.6f}, p1_max={p1_max:.6f} | p2_min={p2_min:.6f}, p2_max={p2_max:.6f}")

    return plot_type, p1, p2, x


def entropy(data: bytes) -> float:
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

# def compare_methods(password: str):
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

        # === Plot-only (Enhanced with normalized x values) ===
    combined_input = password.encode() + salt1 + salt2
    print(f"\n[DEBUG] Combined Input (hex): {combined_input.hex()}")
    print(f"[DEBUG] Combined Input (utf-8 fallback): {combined_input.decode(errors='replace')}")

    x_raw_values = []

    for b in combined_input:
        char = chr(b)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        x_raw_values.append(x)

    # Step 1: Generate secure, per-user range
    user_x_min, user_x_max = generate_user_range(salt1, salt2)

    # Step 2: Normalize all x values into [user_x_min, user_x_max]
    x_values = normalize_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max)

    print(f"\n[DEBUG] User-specific x_min: {user_x_min}, x_max: {user_x_max}")
    print(f"\n[DEBUG] Raw x values        : {x_raw_values}")
    print(f"\n[DEBUG] Normalized x values : {x_values}")

    values = []
    for i, b in enumerate(combined_input):
        char = chr(b)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        result = apply_plot_function(plot_type, p1, p2, x_values[i])
        values.append(result)

    binary_data = struct.pack(f'{len(values)}f', *values)

    print(f"\n[DEBUG] Binary Data (raw float values): {values}")
    print(f"\n[DEBUG] Binary Data (hex): {binary_data.hex()}\n")


    tracemalloc.start()
    start = time.time()
    plot_hash = hashlib.sha256(binary_data).digest()
    plot_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plot_memory_kb = peak / 1024
    plot_entropy = entropy(binary_data)
    plot_len = len(plot_hash)
    plot_verify = (hashlib.sha256(binary_data).digest() == plot_hash)

    results['Plot-only'] = {
        'time': plot_time,
        'entropy': plot_entropy,
        'output_len': plot_len,
        'memory': f"{plot_memory_kb:.3f} KB",
        'verified': plot_verify,
        'hash': plot_hash.hex(),
        'salt': f"{salt1.hex()} + {salt2.hex()}"
    }

    # Plot+Argon2
    tracemalloc.start()
    start = time.time()
    plot_argon_hash = hasher.hash(binary_data)
    plot_argon_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plot_argon_memory_kb = peak / 1024
    plot_argon_entropy = entropy(plot_argon_hash.encode())
    plot_argon_len = len(plot_argon_hash.encode())
    try:
        plot_argon_verify = hasher.verify(plot_argon_hash, binary_data)
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

    # === Plot-only (Enhanced with normalized x values + DEBUG) ===
    combined_input = password.encode() + salt1 + salt2
    print(f"\n[DEBUG] Combined Input (hex): {combined_input.hex()}")
    print(f"[DEBUG] Combined Input (utf-8 fallback): {combined_input.decode(errors='replace')}")

    x_raw_values = []

    print("\n[DEBUG] Character mappings (before normalization):")
    for b in combined_input:
        char = chr(b)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        x_raw_values.append(x)
        print(f"  Char: {repr(char)} | PlotType: {plot_type} | p1: {p1:.6f} | p2: {p2:.6f} | x_raw: {x:.6f}")

    # Step 1: Generate secure, per-user range
    user_x_min, user_x_max = generate_user_range(salt1, salt2)

    # Step 2: Normalize all x values into [user_x_min, user_x_max]
    x_values = normalize_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max)

    print(f"\n[DEBUG] User-specific x_min: {user_x_min}, x_max: {user_x_max}")
    print(f"[DEBUG] Raw x values        : {x_raw_values}")
    print(f"[DEBUG] Normalized x values : {x_values}")

    values = []
    print("\n[DEBUG] Applying plot functions (with normalized x):")
    for i, b in enumerate(combined_input):
        char = chr(b)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        result = apply_plot_function(plot_type, p1, p2, x_values[i])
        values.append(result)
        print(f"  Char: {repr(char)} | PlotType: {plot_type} | p1: {p1:.6f} | p2: {p2:.6f} | "
              f"x_norm: {x_values[i]:.6f} | f(x): {result:.6f}")

    binary_data = struct.pack(f'{len(values)}f', *values)

    print(f"\n[DEBUG] Binary Data (raw float values): {values}")
    print(f"[DEBUG] Binary Data (hex): {binary_data.hex()}\n")

    tracemalloc.start()
    start = time.time()
    plot_hash = hashlib.sha256(binary_data).digest()
    plot_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plot_memory_kb = peak / 1024
    plot_entropy = entropy(binary_data)
    plot_len = len(plot_hash)
    plot_verify = (hashlib.sha256(binary_data).digest() == plot_hash)

    results['Plot-only'] = {
        'time': plot_time,
        'entropy': plot_entropy,
        'output_len': plot_len,
        'memory': f"{plot_memory_kb:.3f} KB",
        'verified': plot_verify,
        'hash': plot_hash.hex(),
        'salt': f"{salt1.hex()} + {salt2.hex()}"
    }

    # Plot+Argon2
    tracemalloc.start()
    start = time.time()
    plot_argon_hash = hasher.hash(binary_data)
    plot_argon_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    plot_argon_memory_kb = peak / 1024
    plot_argon_entropy = entropy(plot_argon_hash.encode())
    plot_argon_len = len(plot_argon_hash.encode())
    try:
        plot_argon_verify = hasher.verify(plot_argon_hash, binary_data)
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
    passwords = ["abc", "password123", "12345678", "AbC4%L98""/?78/IiKkLaDfBn3I0o"]
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

    # Separate plot for Hashing Time
    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
        plt.bar(x + (i - 2.5) * width, all_times[method], width, label=method)
    plt.xticks(x, passwords)
    plt.ylabel("Hashing Time (s)")
    plt.title("Hashing Time Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Separate plot for Entropy
    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
        plt.bar(x + (i - 2.5) * width, all_entropies[method], width, label=method)
    plt.xticks(x, passwords)
    plt.ylabel("Entropy (bits)")
    plt.title("Entropy of Output")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # Separate plot for Output Length
    plt.figure(figsize=(8, 5))
    for i, method in enumerate(methods):
        plt.bar(x + (i - 2.5) * width, all_output_lens[method], width, label=method)
    plt.xticks(x, passwords)
    plt.ylabel("Output Length (bytes)")
    plt.title("Output Length Comparison")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
