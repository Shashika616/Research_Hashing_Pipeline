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
import csv
from statistics import mean, pstdev
from argon2 import low_level as ll
from argon2 import PasswordHasher
from hashlib import sha256
from typing import Tuple, List
from collections import Counter

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

PASSWORDS = [
    "a",
    "password",
    "correcthorsebatterystaple",
    "Tr0ub4dor&3",
    "abc123"
]

MAX_DIMS_RANGE = range(2, 101)
ITERATIONS = 3

salt1 = b"research_salt_1"
salt2 = b"research_salt_2"


# Helper: deterministic salt derivation for repeatable tests
def derive_salts(seed_label: str, attempt: int) -> Tuple[bytes, bytes]:
    """Derive two 8-byte salts deterministically from SECRET_KEY, label and attempt index."""
    label = seed_label.encode() if isinstance(seed_label, str) else seed_label
    base = sha256(SECRET_KEY + label + attempt.to_bytes(4, 'big')).digest()
    return base[:8], base[8:16]

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
    # Convert bytes to uint32 bits
    bits = int.from_bytes(b, byteorder='big')
    sign = (bits >> 31) & 0x1
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    # Clamp exponent to avoid denormals, zeros, Inf, NaN
    orig_exponent = exponent
    if exponent == 0 or exponent == 255:
        exponent = 127  # bias for exponent 0 → safe normal number
    # Clamp mantissa to avoid exact zero
    orig_mantissa = mantissa
    if mantissa == 0:
        mantissa = 1
    safe_bits = (sign << 31) | (exponent << 23) | mantissa
    safe_bytes = safe_bits.to_bytes(4, byteorder='big')
    val = struct.unpack('>f', safe_bytes)[0]

    #  Creative clamp 
    min_val = 1e-12
    max_val = 1e12

    abs_val = abs(val)

    if abs_val < min_val:
        # Small value: clamp using max significant digit * min_val
        digits = get_significant_digits(val, n=5)
        max_digit = max(digits) if digits else 1
        val = math.copysign(max_digit * min_val, val)

    elif abs_val > max_val:
        # Large value: clamp using max significant digit * max_val
        digits = get_significant_digits(val, n=5)
        max_digit = max(digits) if digits else 9
        val = math.copysign(max_digit * max_val, val)

    val = round(val, precision)

    return val

def generate_user_range(salt1: bytes, salt2: bytes) -> Tuple[float, float]:

    def extract_strong_float(hmac_bytes: bytes) -> float:
        val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
        abs_val = abs(val) if val != 0 else 1.0
        exponent = math.log10(abs_val)
        scaled = exponent % 3.0
        res = 10 ** scaled
        return res

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

def normalize_multi_dim_x_values_to_custom_range(xs_list: list[list[float]], x_min: float, x_max: float, method: str) -> list[list[float]]:
    normalized_all = []
    for idx, vec in enumerate(xs_list):
        arr = np.array(vec, dtype=np.float64)
        if arr.size == 0:
            normalized_all.append([])
            continue
        # Apply chosen normalization
        if method == "linear":
            raw_min, raw_max = np.min(arr), np.max(arr)
            if abs(raw_max - raw_min) < 1e-9:
                normed = np.full_like(arr, 0.5)
            else:
                normed = (arr - raw_min) / (raw_max - raw_min)

        elif method == "log":
            normed = np.log1p(np.abs(arr))
            norm_min, norm_max = np.min(normed), np.max(normed)
            if abs(norm_max - norm_min) < 1e-9:
                normed = np.full_like(normed, 0.5)
            else:
                normed = (normed - norm_min) / (norm_max - norm_min)

        elif method == "clipped":
            # clip 1st and 99th percentile
            low, high = np.percentile(arr, [1, 99])
            clipped = np.clip(arr, low, high)
            normed = (clipped - low) / (high - low) if abs(high - low) > 1e-9 else np.full_like(arr, 0.5)

        elif method == "tanh":
            # scale into -1..1 using tanh, then map to 0..1
            normed = np.tanh(arr)
            normed = (normed + 1) / 2
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Map to [x_min, x_max]
        mapped = x_min + normed * (x_max - x_min)
        mapped_rounded = np.round(mapped, 6).tolist()
        normalized_all.append(mapped_rounded)

    return normalized_all

def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
   
    def extract_strong_float(hmac_bytes: bytes) -> float:
        base_val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
        extra_bits = int.from_bytes(hmac_bytes[4:8], byteorder='big')
        frac_rand = (extra_bits % 10000) / 10000.0
        combined = abs(base_val) * (1 + frac_rand)
        scaled = (math.log10(combined + 1e-12) + 6) % 3.0
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

def normalize_param(value: float, param_min: float, param_max: float, method: str = "linear") -> float:
    
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

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes, char_index: int, max_dims: int) -> Tuple[int, float, float, int, List[float]]:

    # plot type
    tag = char.encode('utf-8') + bytes([char_index])  # use i as unique per character instance
    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'type', sha256).digest()
    plot_type = hmac_type[0] % 6
    print("  plot_type -> {}", plot_type)
    # p1 and p2 base values
    hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'p1', sha256).digest()
    raw_p1 = safe_float_from_bytes(hmac_p1[:4])

    hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'p2', sha256).digest()
    raw_p2 = safe_float_from_bytes(hmac_p2[:4])

    # normalize p1 and p2 into user-specific ranges
    p1_min, p1_max = generate_user_param_range(salt1, salt2, b'p1')
    p2_min, p2_max = generate_user_param_range(salt1, salt2, b'p2')
    p1 = normalize_param(raw_p1, p1_min, p1_max, method="linear")
    p2 = normalize_param(raw_p2, p2_min, p2_max, method="linear")

    print("raw_p1 before normalization:", raw_p1)
    print("p1 after normalization:", p1)
    print("raw_p2 before normalization:", raw_p2)
    print("p2 after normalization:", p2)

    # determine n (dimensions) via a dedicated HMAC
    hmac_n = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'n_dims', sha256).digest()
    n = (hmac_n[0] % max_dims) + 1

    # For x_i values, derive using successive HMACs/blocks so each xi is independent
    x_list: List[float] = []
    for i in range(n):
        block_tag = b'x' + bytes([i])
        hmac_xi = hmac.new(SECRET_KEY, salt1 + salt2 + tag + block_tag, sha256).digest()
        xi = safe_float_from_bytes(hmac_xi[:4])
        x_list.append(xi)

    print("  final mapping: plot_type={}, p1={}, p2={}, n={}, x_list={}", plot_type, p1, p2, n, x_list)
    return plot_type, p1, p2, n, x_list

def shannon_entropy(data: bytes) -> float:
    freq = Counter(data)
    total = len(data)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())

def hamming_distance(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))

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

def iterative_plot_transform(password: str, salt1: bytes, salt2: bytes, iterations: int, max_dims: int):
 
    current_input = password.encode()
    all_intermediates = []

    for it in range(1, iterations + 1):
        combined_input = current_input + salt1 + salt2

        x_raw_values = []  

        # Map each byte/char to function and gather raw x vectors
        for idx, b in enumerate(combined_input):
            char = chr(b % 256)
            plot_type, p1, p2, n, x_list = map_char_to_function_with_x(char, salt1, salt2, char_index=idx, max_dims=max_dims)
            x_raw_values.append(x_list)

        # Normalize per-char vectors into the user-specific range
        user_x_min, user_x_max = generate_user_range(salt1, salt2)
        x_values_multi = normalize_multi_dim_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max, "log")

         # Show normalized multi-dim vectors
        for idx, vec in enumerate(x_values_multi):
            print("    normalized vector idx={} -> {}", idx, vec)

        #Re-normalize into a safe mathematical range for log and exp
        x_values_multi = normalize_multi_dim_x_values_to_custom_range(x_values_multi, -700, 700, "log")

        # Show normalized multi-dim vectors
        for idx, vec in enumerate(x_values_multi):
            print("    Math normalized vector idx={} -> {}", idx, vec)


        # Apply plot functions (one scalar result per character)
        values = []
        for i, b in enumerate(combined_input):
            char = chr(b % 256)
            plot_type, p1, p2, n, _ = map_char_to_function_with_x(char, salt1, salt2, char_index=i, max_dims=max_dims)
            x_vec = x_values_multi[i]

            if len(x_vec) < n:
                x_vec = x_vec + [1e-6] * (n - len(x_vec))

            result, func_str = apply_plot_function(plot_type, p1, p2, x_vec)
            # print("    FUNCTION idx={}| char='{}' -> {}", i, char, func_str)
            # print("    OUTPUT   idx={}| char='{}' -> {}", i, char, result)
            values.append(result)


                # === Avalanche Amplification via Permutation & Hash Cascade ===
        hash_accumulator = bytearray(32)  # 32 bytes = 256 bits

        for v in values:
            # Convert float to string, hash it
            h = hashlib.sha256(str(v).encode()).digest()

            # XOR cascade (global mixing)
            for i in range(32):
                hash_accumulator[i] ^= h[i]

            # Randomized permutation: rotate accumulator based on hash byte
            shift_amount = h[0] % 32
            hash_accumulator = bytearray(
                hash_accumulator[shift_amount:] +
                hash_accumulator[:shift_amount]
            )

        # Convert final mixed hash to float values
        # Break into 8 x 4-byte chunks → float32 packable
        extra_floats = []
        for i in range(0, 32, 4):
            chunk = hash_accumulator[i:i+4]
            val = struct.unpack('!f', chunk)[0]  # interpret as float
            if math.isnan(val) or math.isinf(val):
                val = 0.0
            extra_floats.append(val)

        # Append these global-mix floats to the per-character values
        values.extend(extra_floats)

        # Convert to binary with debug
        try:
            binary_data = struct.pack(f'{len(values)}f', *values)
        except struct.error as e:
            # clamp values to safe float range for packing
            safe_vals = []
            for v in values:
                if math.isnan(v) or math.isinf(v):
                    safe_vals.append(0.0)
                else:
                    # packable float32 range approx [-3.4e38, 3.4e38]; our clamp is much narrower
                    safe_vals.append(max(min(v, 1e30), -1e30))
            binary_data = struct.pack(f'{len(safe_vals)}f', *safe_vals)

        all_intermediates.append(binary_data)

        # Next iteration input
        # current_input = binary_data
        current_input = hashlib.sha256(binary_data).digest()

    return current_input

def hamming_distance_bytes(b1, b2):
    if len(b1) < len(b2):
        b1 = b1.ljust(len(b2), b'\x00')
    elif len(b2) < len(b1):
        b2 = b2.ljust(len(b1), b'\x00')
    return sum((x^y).bit_count() for x,y in zip(b1,b2))

def full_pipeline_outputs(password, salt1, salt2, iterations, normalization_method):
    final_binary = iterative_plot_transform(password, salt1, salt2, iterations)
    plot_sha256 = hashlib.sha256(final_binary).digest()
    argon_salt = os.urandom(16)
    argon2_raw = ll.hash_secret_raw(secret=final_binary, salt=argon_salt, time_cost=2, memory_cost=102400,
                                    parallelism=8, hash_len=32, type=ll.Type.ID)
    return final_binary, plot_sha256, argon2_raw

def run_max_dims_experiment():
    rows = []
    baseline_outputs = {}

    for pwd in PASSWORDS:
        baseline_outputs[pwd] = {}

        for max_dims in MAX_DIMS_RANGE:
            start = time.time()

            final_binary = iterative_plot_transform(
                password=pwd,
                salt1=salt1,
                salt2=salt2,
                iterations=ITERATIONS,
                max_dims=max_dims
            )

            elapsed = time.time() - start
            entropy = shannon_entropy(final_binary)

            if not baseline_outputs[pwd]:
                baseline_outputs[pwd]["binary"] = final_binary

            base = baseline_outputs[pwd]["binary"]
            ham = hamming_distance(base, final_binary)
            ham_pct = (ham / (len(base) * 8)) * 100

            rows.append([
                pwd,
                max_dims,
                len(final_binary),
                entropy,
                ham,
                ham_pct,
                elapsed
            ])

    with open("max_dims_experiment.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "password",
            "max_dims",
            "binary_length_bytes",
            "entropy_bits",
            "hamming_distance",
            "avalanche_percent",
            "runtime_seconds"
        ])
        writer.writerows(rows)

    print("Experiment complete → max_dims_experiment.csv")


def run_avalanche_tests(pairs, repetitions=10, iterations=2, normalization_method="log", csv_filename="avalanche_results.csv"):
    headers = ["pair_label","pw1","pw2","method","rep","bits_total","bit_diff","avalanche_percent"]
    rows = []
    for pw1, pw2 in pairs:
        label = f"{pw1}__vs__{pw2}"
        print(f"\n=== Testing pair: {pw1}  vs  {pw2} ===")
        for rep in range(repetitions):
            salt1, salt2 = derive_salts(label, rep)
            plot_bin1, plot_sha1, argon1 = full_pipeline_outputs(pw1, salt1, salt2, iterations, normalization_method)
            plot_bin2, plot_sha2, argon2 = full_pipeline_outputs(pw2, salt1, salt2, iterations, normalization_method)

            # Transformation layer avalanche
            bits_bin = len(plot_bin1) * 8
            diff_bin = hamming_distance_bytes(plot_bin1, plot_bin2)
            perc_bin = diff_bin / bits_bin * 100
            rows.append([label, pw1, pw2, "TransformationLayer", rep, bits_bin, diff_bin, round(perc_bin,4)])

            # Optional intermediate Plot SHA256 avalanche
            bits_sha = len(plot_sha1) * 8
            diff_sha = hamming_distance_bytes(plot_sha1, plot_sha2)
            perc_sha = diff_sha / bits_sha * 100
            rows.append([label, pw1, pw2, "PlotSHA256", rep, bits_sha, diff_sha, round(perc_sha,4)])

            # Full pipeline (Plot+Argon2) avalanche
            bits_arg = len(argon1) * 8
            diff_arg = hamming_distance_bytes(argon1, argon2)
            perc_arg = diff_arg / bits_arg * 100
            rows.append([label, pw1, pw2, "Plot+Argon2", rep, bits_arg, diff_arg, round(perc_arg,4)])

            print(f" rep {rep:02d}  Transformation: {diff_bin}/{bits_bin} bits ({perc_bin:.2f}%)  |  Argon2id: {diff_arg}/{bits_arg} bits ({perc_arg:.2f}%)")
    with open(csv_filename,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nAll results saved to {csv_filename}")


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

     run_max_dims_experiment()