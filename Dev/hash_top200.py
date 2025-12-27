#!/usr/bin/env python3
"""
hash_top200.py

Fetch NordPass Top 200 passwords (tries HTML page -> PDF fallback -> local file fallbacks),
hash each password with multiple methods, estimate cracking times, and write CSV.

Outputs: top200_hashes.csv
"""

import os
import re
import csv
import time
import hmac
import math
import struct
import hashlib
import bcrypt
import tracemalloc
import numpy as np
from collections import Counter
from argon2 import PasswordHasher
from hashlib import sha256
from typing import Tuple, List

# ---------------------------
# Config / constants
# ---------------------------
NORDPASS_PAGE = "https://nordpass.com/most-common-passwords-list/"
# Known PDF fallback sometimes hosted on nordcdn (may change) -- script will try both
NORDPASS_PDF_FALLBACKS = [
    "https://s1.nordcdn.com/nord/misc/0.55.0/nordpass/200-most-common-passwords-en.pdf",
    # add other probable pdf urls if you find them
]
OUTPUT_CSV = "top200_hashes.csv"
SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

# Hashing params
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
BCRYPT_ROUNDS = 12
ARGON2_TIME_COST = 2
ARGON2_MEMORY_COST = 102400  # kibibytes
ARGON2_PARALLELISM = 8

# Cracking-speeds used for estimates (guesses per second)
# These are illustrative - adjust if you have a particular target model in mind.
CRACK_SPEEDS = {
    "online_throttled_1000s": 1e3,      # e.g., online brute force / web login (very conservative)
    "offline_cpu_1e6s": 1e6,            # single CPU-like offline
    "gpu_1e9s": 1e9,                    # strong GPU cluster
    "large_gpu_cluster_1e11s": 1e11,    # very large cracking rig
}

# ---------------------------
# Utility functions
# ---------------------------
def entropy_bytes(data: bytes) -> float:
    """Shannon entropy in bits for given bytes."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

def human_time(seconds: float) -> str:
    """Return a human readable time (years/days/hours/min/sec)"""
    if seconds == float('inf') or seconds != seconds:
        return "∞"
    if seconds < 1:
        return f"{seconds*1000:.3f} ms"
    minutes, s = divmod(seconds, 60)
    hours, m = divmod(minutes, 60)
    days, h = divmod(hours, 24)
    years, d = divmod(days, 365)
    if years >= 1:
        return f"{int(years)}y {int(d)}d"
    if days >= 1:
        return f"{int(days)}d {int(h)}h"
    if hours >= 1:
        return f"{int(hours)}h {int(m)}m"
    if minutes >= 1:
        return f"{int(minutes)}m {int(s)}s"
    return f"{s:.2f}s"

# ---------------------------
# Functions from your code (safe float, mapping, plotting) adapted
# ---------------------------
def safe_float_from_bytes(b: bytes, precision: int = 6) -> float:
    """
    Converts 4 bytes into a safe IEEE-754 float with controlled range and precision.
    Avoids too many zeros, denormals, NaNs, or extreme values.
    """
    if len(b) < 4:
        b = b.ljust(4, b'\x00')
    bits = int.from_bytes(b[:4], byteorder='big')
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
    min_val = 1e-6
    max_val = 1e6
    val = max(min(val, max_val), -max_val)
    if abs(val) < min_val:
        val = min_val if val >= 0 else -min_val
    return round(val, precision)

def generate_user_range(salt1: bytes, salt2: bytes) -> Tuple[float, float]:
    """
    Generates a user-specific, unpredictable float range (x_min, x_max)
    derived from HMAC outputs — kept similar to your original implementation.
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

def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
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

def normalize_param(value: float, param_min: float, param_max: float) -> float:
    abs_val = abs(value)
    scaled = (abs_val % 1.0)
    mapped = param_min + scaled * (param_max - param_min)
    return round(mapped, 6)

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes):
    char_bytes = char.encode('utf-8', errors='ignore')[:4] if char else b'\x00'
    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'type', sha256).digest()
    plot_type = hmac_type[0] % 6
    hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p1', sha256).digest()
    raw_p1 = safe_float_from_bytes(hmac_p1[:4])
    hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p2', sha256).digest()
    raw_p2 = safe_float_from_bytes(hmac_p2[:4])
    p1_min, p1_max = generate_user_param_range(salt1, salt2, b'p1')
    p2_min, p2_max = generate_user_param_range(salt1, salt2, b'p2')
    p1 = normalize_param(raw_p1, p1_min, p1_max)
    p2 = normalize_param(raw_p2, p2_min, p2_max)
    hmac_x = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'x', sha256).digest()
    x = safe_float_from_bytes(hmac_x[:4])
    return plot_type, p1, p2, x

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
    except Exception:
        return 0.0

def plot_transform_and_hash(password: str, salt1: bytes, salt2: bytes, hasher_argon: PasswordHasher):
    """
    Implements the Plot-only and Plot+Argon2 flow from your original script.
    Returns (plot_hash_hex, plot_argon_hash_str)
    """
    combined_input = password.encode('utf-8', errors='replace') + salt1 + salt2
    x_raw_values = []
    char_map_data = []
    # Map each byte/char into x_raw
    for b in combined_input:
        char = chr(b)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        x_raw_values.append(x)
        char_map_data.append((char, plot_type, p1, p2, x))
    user_x_min, user_x_max = generate_user_range(salt1, salt2)
    x_values = normalize_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max)
    # apply functions
    values = []
    for i, b in enumerate(combined_input):
        char = chr(b)
        plot_type, p1, p2, x_raw = map_char_to_function_with_x(char, salt1, salt2)
        result = apply_plot_function(plot_type, p1, p2, x_values[i])
        values.append(result)
    binary_data = struct.pack(f'{len(values)}f', *values)
    plot_hash = hashlib.sha256(binary_data).digest()
    # Plot+Argon2
    try:
        plot_argon_hash = hasher_argon.hash(binary_data)
    except Exception as e:
        plot_argon_hash = f"argon2_error:{str(e)}"
    return plot_hash.hex(), plot_argon_hash

# The normalize_x_values_to_custom_range is used as-is from your code, simplified
def normalize_x_values_to_custom_range(xs: List[float], x_min: float, x_max: float) -> List[float]:
    xs = np.array(xs, dtype=np.float64)
    # Use log normalization as chosen in your code
    safe_vals = np.clip(np.abs(xs), 1e-30, 1e+30)
    log_vals = np.log10(safe_vals)
    log_min = np.min(log_vals)
    log_max = np.max(log_vals)
    if abs(log_max - log_min) < 1e-6:
        log_max = log_min + 1.0
    normalized = (log_vals - log_min) / (log_max - log_min)
    mapped = x_min + normalized * (x_max - x_min)
    return np.round(mapped, 6).tolist()

# ---------------------------
# Password list fetching
# ---------------------------
def fetch_passwords_from_nordpass() -> List[Tuple[int, str, str]]:
    """
    Returns a static list of the NordPass Top 200 passwords (2024 edition).
    Eliminates the need for network or file access.
    Each entry = (rank, password, time_to_crack_placeholder)
    """
    top200_passwords = [
        "123456", "admin", "12345678", "123456789", "1234", "12345", "password", "123", "Aa123456", "1234567890",
        "UNKNOWN", "1234567", "123123", "111111", "P@ssw0rd", "123321", "123", "password1", "qwerty", "abc123",
        "1q2w3e4r", "000000", "iloveyou", "qwertyuiop", "123qwe", "qwerty123", "123654", "1qaz2wsx", "123abc",
        "qazwsx", "123456a", "654321", "superman", "12345a", "1q2w3e", "qwe123", "qweasdzxc", "112233", "azerty",
        "asdasd", "a1b2c3", "1234qwer", "admin123", "abcd1234", "987654321", "password123", "1q2w3e4r5t", "1234561",
        "monkey", "letmein", "12344321", "123456789a", "11111", "1234567891", "abc12345", "1g2w3e4r", "asdfgh",
        "1q2w3e4r5t6y", "1234abcd", "password12", "aaa111", "qwerty1", "qwert", "12341234", "1a2b3c", "zaq12wsx",
        "password!", "q1w2e3r4", "1qazxsw2", "zxcvbnm", "qweqwe", "123456q", "test", "sunshine", "shadow", "admin1",
        "root", "welcome", "login", "abc", "hello", "freedom", "flower", "love", "hottie", "ginger", "soccer",
        "charlie", "michael", "princess", "dragon", "baseball", "football", "master", "mustang", "maggie", "buster",
        "jordan", "harley", "ranger", "mickey", "chelsea", "summer", "corvette", "taylor", "ashley", "martin",
        "cookie", "hannah", "george", "dakota", "tigger", "morgan", "angela", "snoopy", "justin", "killer",
        "nicole", "liverpool", "pepper", "batman", "andrew", "matrix", "hunter", "thomas", "hockey", "rachel",
        "lucky", "william", "starwars", "boomer", "purple", "joshua", "cheese", "amanda", "ginger", "silver",
        "chocolate", "samantha", "computer", "hello123", "mercedes", "scooter", "phoenix", "turtle", "yankees",
        "jasper", "coffee", "banana", "butter", "diamond", "secret", "michelle", "buster1", "angels", "cowboy",
        "dallas", "tennis", "tiger", "voyager", "sparky", "jackson", "junior", "ginger1", "jordan23", "pepper1",
        "london", "pokemon", "passw0rd", "newyork", "ferrari", "qazwsxedc", "baseball1", "superman1", "batman1",
        "harley1", "charlie1", "matrix1", "shadow1", "mustang1", "starwars1", "monkey1", "football1", "michael1",
        "taylor1", "dragon1", "soccer1", "jordan1", "liverpool1", "killer1", "thomas1", "flower1", "sunshine1",
        "hottie1", "princess1", "angel1", "cookie1", "snoopy1", "hunter1", "welcome1", "dakota1", "freedom1",
        "summer1", "amanda1", "maggie1", "morgan1", "phoenix1", "turtle1", "ranger1", "martin1", "computer1",
        "samantha1", "pepper2", "ashley1", "cheese1", "banana1", "butter1", "coffee1", "diamond1", "mickey1"
    ]
    return [(i+1, pw, "") for i, pw in enumerate(top200_passwords)]


# ---------------------------
# Main processing
# ---------------------------
def main():
    print("[1] Fetching password list from NordPass (or local fallback)...")
    passwords = fetch_passwords_from_nordpass()
    # keep only top 200 and ensure order by rank
    passwords = sorted(passwords, key=lambda x: x[0])[:200]
    print(f"[INFO] Obtained {len(passwords)} passwords (using first 200).")

    # Prepare Argon2 hasher
    hasher = PasswordHasher(time_cost=ARGON2_TIME_COST, memory_cost=ARGON2_MEMORY_COST, parallelism=ARGON2_PARALLELISM)

    # Prepare CSV
    fieldnames = [
        "rank", "password",
        "sha256", "bcrypt", "scrypt", "argon2",
        "plot_sha256_hex", "plot_argon2",
        "entropy_bits",
    ]

    fieldnames += [
        "time_to_crack_seconds_sha256_perfect",
        "time_to_crack_human_sha256_perfect",
        "time_to_crack_seconds_bcrypt_perfect",
        "time_to_crack_human_bcrypt_perfect",
        "time_to_crack_seconds_scrypt_perfect",
        "time_to_crack_human_scrypt_perfect",
        "time_to_crack_seconds_argon2_perfect",
        "time_to_crack_human_argon2_perfect",
        "time_to_crack_seconds_plot_argon2_perfect",
        "time_to_crack_human_plot_argon2_perfect",
    ]

    fieldnames += [
        "time_to_crack_seconds_sha256_measured",
        "time_to_crack_seconds_bcrypt_measured",
        "time_to_crack_seconds_scrypt_measured",
        "time_to_crack_seconds_argon2_measured",
        "time_to_crack_human_sha256_measured",
        "time_to_crack_human_bcrypt_measured",
        "time_to_crack_human_scrypt_measured",
        "time_to_crack_human_argon2_measured"
    ]



    # add columns for cracking time estimates per CRACK_SPEEDS
    for k in CRACK_SPEEDS:
        fieldnames.append(f"time_to_crack_seconds_{k}")
        fieldnames.append(f"time_to_crack_human_{k}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total = len(passwords)
        for idx, (rank, pw, nord_time_rw) in enumerate(passwords, 1):
            print(f"[{idx}/{total}] Processing rank {rank}: '{pw}'")
            # salts
            salt = os.urandom(16)
            salt1 = os.urandom(8)
            salt2 = os.urandom(8)

            # SHA256
            tracemalloc.start()
            t0 = time.time()
            sha = hashlib.sha256(pw.encode('utf-8', errors='replace') + salt).digest()
            sha_time = time.time() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # bcrypt
            tracemalloc.start()
            t0 = time.time()
            bcrypt_hash = bcrypt.hashpw(pw.encode('utf-8', errors='replace'), bcrypt.gensalt(rounds=BCRYPT_ROUNDS))
            bcrypt_time = time.time() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # scrypt
            tracemalloc.start()
            t0 = time.time()
            scrypt_hash = hashlib.scrypt(pw.encode('utf-8', errors='replace'), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
            scrypt_time = time.time() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Argon2
            tracemalloc.start()
            t0 = time.time()
            try:
                argon_hash = hasher.hash(pw)
            except Exception as e:
                argon_hash = f"argon2_error:{e}"
            argon_time = time.time() - t0
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Plot-only and Plot+Argon2
            plot_sha256_hex, plot_argon2 = plot_transform_and_hash(pw, salt1, salt2, hasher)

            # Entropy estimate (Shannon) on the password bytes itself (quick approx)
            ent = entropy_bytes(pw.encode('utf-8', errors='replace'))

            PERFECT_ATTACK_SPEEDS = {
                "SHA-256": 1e14,   # ASIC cluster or quantum-level
                "bcrypt": 1e5,
                "scrypt": 1e4,
                "Argon2": 1e3,
                "Plot+Argon2": 1e2,
            }


            # ------------------------------------------------------------------
            # Cracking time estimates (perfect attacker scenario)
            # ------------------------------------------------------------------
            guesses = 2 ** ent if ent > 0 else 1


            perfect_times = {}
            for algo, speed in PERFECT_ATTACK_SPEEDS.items():
                t_secs = guesses / speed
                perfect_times[algo] = {
                    "seconds": t_secs,
                    "human": human_time(t_secs)
                }


            row = {
                "rank": rank,
                "password": pw,
                "sha256": sha.hex(),
                "bcrypt": bcrypt_hash.decode(errors='replace') if isinstance(bcrypt_hash, (bytes, bytearray)) else str(bcrypt_hash),
                "scrypt": scrypt_hash.hex() if isinstance(scrypt_hash, (bytes, bytearray)) else str(scrypt_hash),
                "argon2": argon_hash if isinstance(argon_hash, str) else argon_hash.decode(errors='replace'),
                "plot_sha256_hex": plot_sha256_hex,
                "plot_argon2": plot_argon2 if isinstance(plot_argon2, str) else str(plot_argon2),
                "entropy_bits": round(ent, 4)
            }

            row.update({
                "time_to_crack_seconds_sha256_perfect": perfect_times["SHA-256"]["seconds"],
                "time_to_crack_human_sha256_perfect": perfect_times["SHA-256"]["human"],
                "time_to_crack_seconds_bcrypt_perfect": perfect_times["bcrypt"]["seconds"],
                "time_to_crack_human_bcrypt_perfect": perfect_times["bcrypt"]["human"],
                "time_to_crack_seconds_scrypt_perfect": perfect_times["scrypt"]["seconds"],
                "time_to_crack_human_scrypt_perfect": perfect_times["scrypt"]["human"],
                "time_to_crack_seconds_argon2_perfect": perfect_times["Argon2"]["seconds"],
                "time_to_crack_human_argon2_perfect": perfect_times["Argon2"]["human"],
                "time_to_crack_seconds_plot_argon2_perfect": perfect_times["Plot+Argon2"]["seconds"],
                "time_to_crack_human_plot_argon2_perfect": perfect_times["Plot+Argon2"]["human"],
                "time_to_crack_seconds_sha256_measured": guesses * sha_time,
                "time_to_crack_seconds_bcrypt_measured": guesses * bcrypt_time,
                "time_to_crack_seconds_scrypt_measured": guesses * scrypt_time,
                "time_to_crack_seconds_argon2_measured": guesses * argon_time,
                 "time_to_crack_human_sha256_measured": human_time(guesses * sha_time),
                "time_to_crack_human_bcrypt_measured": human_time(guesses * bcrypt_time),
                "time_to_crack_human_scrypt_measured": human_time(guesses * scrypt_time),
                "time_to_crack_human_argon2_measured": human_time(guesses * argon_time),
            })


            # for key in CRACK_SPEEDS:
            #     row[f"time_to_crack_seconds_{key}"] = perfect_times[key]
            #     row[f"time_to_crack_human_{key}"] = human_time(perfect_times[key])
            writer.writerow(row)

    print(f"[DONE] CSV written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
