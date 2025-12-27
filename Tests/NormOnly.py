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
from sympy import sstr


DEBUG = True

open("debug_log.txt", "w").close()

LOG_FILE = "debug_log.txt"

def log(msg):
    with open("debug_log.txt", "a", encoding="utf-8") as f:  # <- add encoding
        f.write(sstr(msg) + "\n")

def dbg(msg: str, /, *args):

    if DEBUG:
        formatted = msg.format(*args)
        print(formatted)           # still prints on screen
        log(formatted)             # also writes to log file

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

def get_significant_digits(val, n=5):
    # Return the first n significant digits of a number as a list of integers.
    if val == 0:
        return [0]
    exponent = math.floor(math.log10(abs(val)))
    scaled = abs(val) / (10 ** exponent)  # scale to [1,10)
    digits = []
    for _ in range(n):
        digit = int(scaled)
        digits.append(digit)
        scaled = (scaled - digit) * 10
    return digits

def safe_float_from_bytes(b: bytes, precision: int = 12) -> float:
    if len(b) < 4:
        # pad if necessary
        b = b.ljust(4, b'\x00')
    dbg("[safe_float_from_bytes] input bytes hex: {}", b.hex())
    # Convert bytes to uint32 bits
    bits = int.from_bytes(b, byteorder='big')
    dbg("[safe_float_from_bytes] raw bits: {:032b}", bits)
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    dbg("  sign={}, exponent={}, mantissa={}", sign, exponent, mantissa)
    # Clamp exponent to avoid denormals, zeros, Inf, NaN
    orig_exponent = exponent
    if exponent == 0 or exponent == 255:
        exponent = 127  # bias for exponent 0 → safe normal number
        dbg("  exponent clamped {} -> {}", orig_exponent, exponent)
    # Clamp mantissa to avoid exact zero
    orig_mantissa = mantissa
    if mantissa == 0:
        mantissa = 1
        dbg("  mantissa clamped {} -> {}", orig_mantissa, mantissa)
    safe_bits = (sign << 31) | (exponent << 23) | mantissa
    safe_bytes = safe_bits.to_bytes(4, byteorder='big')
    dbg("  safe bits: {:032b}, safe bytes hex: {}", safe_bits, safe_bytes.hex())
    val = struct.unpack('>f', safe_bytes)[0]
    dbg("  unpacked float before magnitude clamp: {}", val)

    # ---- Creative clamp ----
    min_val = 1e-12
    max_val = 1e12

    abs_val = abs(val)

    if abs_val < min_val:
        # Small value: clamp using max significant digit * min_val
        digits = get_significant_digits(val, n=5)
        max_digit = max(digits) if digits else 1
        val = math.copysign(max_digit * min_val, val)
        dbg("  magnitude creatively clamped for small value -> {}", val)

    elif abs_val > max_val:
        # Large value: clamp using max significant digit * max_val
        digits = get_significant_digits(val, n=5)
        max_digit = max(digits) if digits else 9
        val = math.copysign(max_digit * max_val, val)
        dbg("  magnitude creatively clamped for large value -> {}", val)

    # Values within range remain unchanged
    # Round final value
    val = round(val, precision)
    dbg("  final safe float (rounded): {}", val)


    return val

def generate_user_range(salt1: bytes, salt2: bytes) -> Tuple[float, float]:

    def extract_strong_float(hmac_bytes: bytes) -> float:
        dbg("[generate_user_range.extract] hmac bytes hex: {}", hmac_bytes.hex())
        val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
        dbg("  extracted safe_float_from_bytes -> {}", val)
        abs_val = abs(val) if val != 0 else 1.0
        exponent = math.log10(abs_val)
        dbg("  log10(abs_val) -> {}", exponent)
        scaled = exponent % 3.0
        dbg("  scaled (mod 3.0) -> {}", scaled)
        res = 10 ** scaled
        dbg("  returned extract_strong_float -> {}", res)
        return res

    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + b'unpredictable_x_min', sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + b'unpredictable_x_max', sha256).digest()
    dbg("[generate_user_range] hmac_min: {}, hmac_max: {}", hmac_min.hex(), hmac_max.hex())

    min_val = extract_strong_float(hmac_min)
    max_val = extract_strong_float(hmac_max)
    dbg("  raw min_val={}, max_val={}", min_val, max_val)

    if max_val < min_val:
        min_val, max_val = max_val, min_val
        dbg("  swapped min/max -> min_val={}, max_val={}", min_val, max_val)

    if abs(max_val - min_val) < 1.0:
        separation_hmac = hmac.new(SECRET_KEY, salt1 + salt2 + b'separation', sha256).digest()
        dbg("  separation_hmac: {}", separation_hmac.hex())
        separation = round(safe_float_from_bytes(separation_hmac[:4]) % 10.0, 6)
        dbg("  separation raw -> {}", separation)
        max_val = round(min_val + separation + 1.5, 6)
        dbg("  adjusted max_val -> {}", max_val)

    dbg("[generate_user_range] returning (min,max) = ({},{})", round(min_val,6), round(max_val,6))
    return round(min_val, 6), round(max_val, 6)

def normalize_multi_dim_x_values_to_custom_range(xs_list: list[list[float]], x_min: float, x_max: float, method: str = "linear") -> list[list[float]]:
    dbg("[normalize_multi_dim] method={}, x_min={}, x_max={}", method, x_min, x_max)
    normalized_all = []
    for idx, vec in enumerate(xs_list):
        dbg("  [normalize_multi_dim] char_idx={} raw_vec={}", idx, vec)
        arr = np.array(vec, dtype=np.float64)
        if arr.size == 0:
            dbg("    empty vector -> normalized as []")
            normalized_all.append([])
            continue
        # Apply chosen normalization
        if method == "linear":
            raw_min, raw_max = np.min(arr), np.max(arr)
            if abs(raw_max - raw_min) < 1e-9:
                normed = np.full_like(arr, 0.5)
                dbg("    uniform vector -> normed midpoint: {}", normed)
            else:
                normed = (arr - raw_min) / (raw_max - raw_min)
                dbg("    linear normalized (0..1) -> {}", normed.tolist())

        elif method == "log":
            normed = np.log1p(np.abs(arr))
            norm_min, norm_max = np.min(normed), np.max(normed)
            if abs(norm_max - norm_min) < 1e-9:
                normed = np.full_like(normed, 0.5)
            else:
                normed = (normed - norm_min) / (norm_max - norm_min)
            dbg("    log normalized (0..1) -> {}", normed.tolist())

        elif method == "clipped":
            # clip 1st and 99th percentile
            low, high = np.percentile(arr, [1, 99])
            clipped = np.clip(arr, low, high)
            normed = (clipped - low) / (high - low) if abs(high - low) > 1e-9 else np.full_like(arr, 0.5)
            dbg("    clipped & linear normalized (0..1) -> {}", normed.tolist())

        elif method == "tanh":
            # scale into -1..1 using tanh, then map to 0..1
            normed = np.tanh(arr)
            normed = (normed + 1) / 2
            dbg("    tanh normalized (0..1) -> {}", normed.tolist())

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Map to [x_min, x_max]
        mapped = x_min + normed * (x_max - x_min)
        mapped_rounded = np.round(mapped, 6).tolist()
        dbg("    mapped to [{}..{}] -> {}", x_min, x_max, mapped_rounded)
        normalized_all.append(mapped_rounded)

    dbg("[normalize_multi_dim] completed normalization for {} vectors", len(xs_list))
    return normalized_all

def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
   
    def extract_strong_float(hmac_bytes: bytes) -> float:
        dbg("[generate_user_param_range.extract] hmac bytes hex: {}", hmac_bytes.hex())
        base_val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
        dbg("  base_val -> {}", base_val)
        extra_bits = int.from_bytes(hmac_bytes[4:8], byteorder='big')
        dbg("  extra_bits (int) -> {}", extra_bits)
        frac_rand = (extra_bits % 10000) / 10000.0
        dbg("  frac_rand -> {}", frac_rand)
        combined = abs(base_val) * (1 + frac_rand)
        dbg("  combined -> {}", combined)
        scaled = (math.log10(combined + 1e-12) + 6) % 3.0
        res = round(10 ** scaled, 6)
        dbg("  scaled exponent -> {}, result -> {}", scaled, res)
        return res

    hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_min', sha256).digest()
    hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_max', sha256).digest()
    dbg("[generate_user_param_range] label={} hmac_min={} hmac_max={}", label, hmac_min.hex(), hmac_max.hex())

    min_val = extract_strong_float(hmac_min)
    max_val = extract_strong_float(hmac_max)
    dbg("  raw param range -> min={}, max={}", min_val, max_val)

    if max_val < min_val:
        min_val, max_val = max_val, min_val
        dbg("  swapped param range -> min={}, max={}", min_val, max_val)

    if abs(max_val - min_val) < 5.0:
        sep_hmac = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_sep', sha256).digest()
        dbg("  sep_hmac -> {}", sep_hmac.hex())
        separation = (int.from_bytes(sep_hmac[:2], 'big') % 50) / 10.0
        dbg("  separation -> {}", separation)
        max_val = round(min_val + 5.0 + separation, 6)
        dbg("  adjusted max_val -> {}", max_val)

    return round(min_val, 6), round(max_val, 6)

def normalize_param(value: float, param_min: float, param_max: float, method: str = "linear") -> float:
    
    dbg("[normalize_param] value={}, param_min={}, param_max={}, method={}", value, param_min, param_max, method)
    x = np.array([value], dtype=np.float64)

    def linear_norm(x):
        scaled = abs(x) % 1.0
        dbg("  [linear_norm] scaled -> {}", scaled)
        return param_min + scaled * (param_max - param_min)

    def log_norm(x):
        safe_val = np.clip(np.abs(x), 1e-30, 1e30)
        dbg("  [log_norm] safe_val -> {}", safe_val)
        log_val = np.log10(safe_val)
        log_min, log_max = log_val, log_val
        if abs(log_max - log_min) < 1e-6:
            log_max = log_min + 1.0
        normed = (log_val - log_min) / (log_max - log_min)
        dbg("  [log_norm] normed -> {}", normed)
        return param_min + normed * (param_max - param_min)

    def clipped_norm(x):
        lower, upper = -1.0, 1.0
        clipped = np.clip(x, lower, upper)
        dbg("  [clipped_norm] clipped -> {}", clipped)
        normed = (clipped - lower) / (upper - lower)
        dbg("  [clipped_norm] normed -> {}", normed)
        return param_min + normed * (param_max - param_min)

    def tanh_norm(x):
        scale = 1e10
        squashed = np.tanh(x / scale)
        dbg("  [tanh_norm] squashed -> {}", squashed)
        tanh_min, tanh_max = squashed, squashed
        if abs(tanh_max - tanh_min) < 1e-6:
            tanh_max = tanh_min + 1.0
        normed = (squashed - tanh_min) / (tanh_max - tanh_min)
        dbg("  [tanh_norm] normed -> {}", normed)
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
    dbg("  normalized param -> {}", normalized)
    return round(normalized, 6)

# ---  apply_plot_function ---

def apply_plot_function(plot_type: int, p1: float, p2: float, x_list: List[float]) -> Tuple[float, str]:
    
    n = len(x_list)
    expr_val = 0.0
    expr_str_parts = []

    if plot_type == 0:
        for i in range(n):
            for j in range(n):
                term_val = ((-1)**(i+j)) * p1 * x_list[i] * x_list[j] - p2 * x_list[i]**(j+2)
                expr_val += term_val
                expr_str_parts.append(f"({(-1)**(i+j)}*{p1}*x{i+1}*x{j+1} - {p2}*x{i+1}^{j+2})")

    elif plot_type == 1:
        for i in range(n):
            term_val = p1 * math.sin(x_list[i]) + p2 * x_list[i]**3
            expr_val += term_val
            expr_str_parts.append(f"({p1}*sin(x{i+1}) + {p2}*x{i+1}^3)")

    elif plot_type == 2:
        for i in range(n):
            term_val = p1 * math.log(1 + abs(x_list[i])) + p2 * x_list[i]**2
            expr_val += term_val
            expr_str_parts.append(f"({p1}*log(1+|x{i+1}|) + {p2}*x{i+1}^2)")

    elif plot_type == 3:
        for i in range(n):
            term_val = p1 * math.exp(x_list[i]) - p2 * x_list[i]**2
            expr_val += term_val
            expr_str_parts.append(f"({p1}*exp(x{i+1}) - {p2}*x{i+1}^2)")

    elif plot_type == 4:
        for i in range(n):
            term_val = p1 * x_list[i]/(1 + x_list[i]**2) - p2 * x_list[i]**3
            expr_val += term_val
            expr_str_parts.append(f"({p1}*x{i+1}/(1+x{i+1}^2) - {p2}*x{i+1}^3)")

    elif plot_type == 5:
        for i in range(n):
            for j in range(n):
                term_val = ((-1)**i)*p1*math.sin(x_list[i]) + p2*math.exp(x_list[j]) - math.log(1 + abs(x_list[i]*x_list[j]))
                expr_val += term_val
                expr_str_parts.append(f"({(-1)**i}*{p1}*sin(x{i+1}) + {p2}*exp(x{j+1}) - log(1+|x{i+1}*x{j+1}|))")

    func_repr = " + ".join(expr_str_parts)
    return expr_val, func_repr

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes, char_index: int, max_dims: int = 8) -> Tuple[int, float, float, int, List[float]]:
   
    char_bytes = char.encode('utf-8')
    dbg("[map_char_to_function_with_x] char='{}' bytes={}", char, char_bytes.hex())
    # plot type
    tag = char.encode('utf-8') + bytes([char_index])  # use i as unique per character instance
    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'type', sha256).digest()
    dbg("  hmac_type hex: {}", hmac_type.hex())
    plot_type = hmac_type[0] % 6
    dbg("  plot_type -> {}", plot_type)
    print("  plot_type -> {}", plot_type)
    # p1 and p2 base values
    hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'p1', sha256).digest()
    dbg("  hmac_p1 hex: {}", hmac_p1.hex())
    raw_p1 = safe_float_from_bytes(hmac_p1[:4])
    dbg("  raw_p1 -> {}", raw_p1)

    hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'p2', sha256).digest()
    dbg("  hmac_p2 hex: {}", hmac_p2.hex())
    raw_p2 = safe_float_from_bytes(hmac_p2[:4])
    dbg("  raw_p2 -> {}", raw_p2)

    # normalize p1 and p2 into user-specific ranges
    p1_min, p1_max = generate_user_param_range(salt1, salt2, b'p1')
    p2_min, p2_max = generate_user_param_range(salt1, salt2, b'p2')
    dbg("  p1 range: {} - {}, p2 range: {} - {}", p1_min, p1_max, p2_min, p2_max)
    p1 = normalize_param(raw_p1, p1_min, p1_max, method="linear")
    p2 = normalize_param(raw_p2, p2_min, p2_max, method="linear")
    dbg("  normalized p1={}, p2={}", p1, p2)

    print("raw_p1 before normalization:", raw_p1)
    print("p1 after normalization:", p1)
    print("raw_p2 before normalization:", raw_p2)
    print("p2 after normalization:", p2)

    # determine n (dimensions) via a dedicated HMAC
    hmac_n = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'n_dims', sha256).digest()
    dbg("  hmac_n hex: {}", hmac_n.hex())
    n = (hmac_n[0] % max_dims) + 1
    dbg("  mapped n_dims -> {}", n)

    # For x_i values, derive using successive HMACs/blocks so each xi is independent
    x_list: List[float] = []
    for i in range(n):
        block_tag = b'x' + bytes([i])
        hmac_xi = hmac.new(SECRET_KEY, salt1 + salt2 + tag + block_tag, sha256).digest()
        dbg("    hmac_xi[{}] hex: {}", i, hmac_xi.hex())
        xi = safe_float_from_bytes(hmac_xi[:4])
        dbg("    xi[{}] -> {}", i, xi)
        x_list.append(xi)

    dbg("  final mapping: plot_type={}, p1={}, p2={}, n={}, x_list={}", plot_type, p1, p2, n, x_list)
    print("  final mapping: plot_type={}, p1={}, p2={}, n={}, x_list={}", plot_type, p1, p2, n, x_list)
    return plot_type, p1, p2, n, x_list

def entropy(data: bytes) -> float:
    counts = Counter(data)
    total = len(data)
    # avoid division by zero
    if total == 0:
        return 0.0
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
    dbg("[iterative_plot_transform] password='{}' iterations={}", password, iterations)
    current_input = password.encode()
    all_intermediates = []

    for it in range(1, iterations + 1):
        dbg("\n[iter] Starting iteration {}", it)
        combined_input = current_input + salt1 + salt2
        dbg("  combined_input (len={}): {}", len(combined_input), combined_input.hex())

        # Step 1: Map chars once and collect raw x values
        per_char_params = []
        x_raw_values = []
        for idx, b in enumerate(combined_input):
            char = chr(b % 256)
            plot_type, p1, p2, n, x_list = map_char_to_function_with_x(char, salt1, salt2, char_index=idx)
            dbg("    mapped char_index={} char='{}' -> plot_type={}, p1={}, p2={}, n={}, x_list={}",
                idx, char, plot_type, p1, p2, n, x_list)
            per_char_params.append((plot_type, p1, p2, n))
            x_raw_values.append(x_list)

        # Step 2: Normalize all raw x values into user-specific range
        user_x_min, user_x_max = generate_user_range(salt1, salt2)
        dbg("  user_x_min={}, user_x_max={}", user_x_min, user_x_max)
        x_values_multi = normalize_multi_dim_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max, "log")

        # Step 3: Apply plot functions using normalized x values
        values = []
        for i, (plot_type, p1, p2, n) in enumerate(per_char_params):
            x_vec = x_values_multi[i]
            if len(x_vec) < n:
                x_vec += [1e-6] * (n - len(x_vec))

            dbg("    normalized x_vec idx={} -> {}", i, x_vec)
            result, func_str = apply_plot_function(plot_type, p1, p2, x_vec)
            dbg("    FUNCTION idx={}| char='{}' -> {}", i, chr(combined_input[i] % 256), func_str)
            dbg("    OUTPUT   idx={}| char='{}' -> {}", i, chr(combined_input[i] % 256), result)
            values.append(result)

        # Step 4: Convert results to binary safely
        try:
            binary_data = struct.pack(f'{len(values)}f', *values)
        except struct.error as e:
            dbg("  struct.pack failed: {}. Clamping values...", e)
            safe_vals = [max(min(v, 1e30), -1e30) if not (math.isnan(v) or math.isinf(v)) else 0.0 for v in values]
            binary_data = struct.pack(f'{len(safe_vals)}f', *safe_vals)

        all_intermediates.append(binary_data)
        current_input = binary_data  # feed binary to next iteration

    return current_input, all_intermediates, x_values_multi


def compare_methods(password: str):
    dbg("\n[compare_methods] password='{}'", password)
    salt = os.urandom(16)
    salt1 = os.urandom(8)
    salt2 = os.urandom(8)
    dbg("  salts: salt={}, salt1={}, salt2={}", salt.hex(), salt1.hex(), salt2.hex())

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
    dbg("  SHA256 -> time: {}, mem_kb: {}, entropy: {}, len: {}", sha_time, sha_memory_kb, sha_entropy, sha_len)

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
    dbg("  bcrypt -> time: {}, mem_kb: {}, entropy: {}, len: {}", bcrypt_time, bcrypt_memory_kb, bcrypt_entropy, bcrypt_len)

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
    dbg("  scrypt -> time: {}, mem_kb: {}, entropy: {}, len: {}", scrypt_time, scrypt_memory_kb, scrypt_entropy, scrypt_len)

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
    except Exception as e:
        dbg("  Argon2 verify raised: {}", e)
        argon_verify = False
    dbg("  Argon2 -> time: {}, mem_kb: {}, entropy: {}, len: {}", argon_time, argon_memory_kb, argon_entropy, argon_len)

    results['Argon2'] = {
        'time': argon_time,
        'entropy': argon_entropy,
        'output_len': argon_len,
        'memory': f"{argon_memory_kb:.3f} KB",
        'verified': argon_verify,
        'hash': argon_hash,
        'salt': "Embedded in hash string"
    }

    #  Iterative Plot-only / Plot+Argon2 
    iterations = 2  # you can adjust this

    final_binary, all_intermediates, x_values_multi = iterative_plot_transform(password, salt1, salt2, iterations)

    # Full intermediate binary data
    dbg("\n[DEBUG] Full intermediate binaries:")
    for i, bdata in enumerate(all_intermediates):
        dbg("[Iteration {}] Binary len={} hex:\n{}", i+1, len(bdata), bdata.hex())

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
    dbg("  Plot-only -> time: {}, mem_kb: {}, entropy: {}, len: {}", plot_time, plot_memory_kb, plot_entropy, plot_len)

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
    except Exception as e:
        dbg("  plot_argon verify exception: {}", e)
        plot_argon_verify = False
    dbg("  Plot+Argon2 -> time: {}, mem_kb: {}, entropy: {}, len: {}", plot_argon_time, plot_argon_memory_kb, plot_argon_entropy, plot_argon_len)

    results['Plot+Argon2'] = {
        'time': plot_argon_time,
        'entropy': plot_argon_entropy,
        'output_len': plot_argon_len,
        'memory': f"{plot_argon_memory_kb:.3f} KB",
        'verified': plot_argon_verify,
        'hash': plot_argon_hash,
        'salt': f"{salt1.hex()} + {salt2.hex()}"
    }

    return results, x_values_multi

def print_results(results, password):
    print(f"\n=== Results for password: '{password}' ===")
    print(f"{'Method':<12} | {'Time (s)':<10} | {'Entropy':<10} | {'Output Len':<10} | {'Memory Usage':<25} | {'Verified':<8} | {'Sample Hash (truncated)'}")
    print("-"*110)
    for method, data in results.items():
        truncated_hash = (data['hash'][:40] + '...') if isinstance(data['hash'], str) and len(data['hash']) > 40 else (data['hash'] if isinstance(data['hash'], str) else data['hash'].hex()[:40] + '...')
        print(f"{method:<12} | {data['time']:<10.6f} | {data['entropy']:<10.4f} | {data['output_len']:<10} | {data['memory']:<25} | {str(data['verified']):<8} | {truncated_hash}")
        print(f"{'':<12}   Full Hash: {data['hash']}")
        print(f"{'':<12}   Salt     : {data['salt']}\n")

if __name__ == "__main__":
    passwords = ["aaa"]
    methods = ["SHA256", "bcrypt", "scrypt", "Argon2", "Plot-only", "Plot+Argon2"]

    all_times = {m: [] for m in methods}
    all_entropies = {m: [] for m in methods}
    all_output_lens = {m: [] for m in methods}

    for pwd in passwords:
        results, x_values_multi = compare_methods(pwd)
        print_results(results, pwd)

        for method in methods:
            all_times[method].append(results[method]['time'])
            all_entropies[method].append(results[method]['entropy'])
            all_output_lens[method].append(results[method]['output_len'])
 
    plot_results_plotly(passwords, all_times, all_entropies, all_output_lens, methods)
