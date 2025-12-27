#!/usr/bin/env python3
"""
Enhanced version — supports iterative transformation:
- Each round applies the same plot transformation pipeline again on its previous binary result.
- After N rounds, the final binary output is hashed with Argon2.
"""

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

# ==============================================
# Configuration
# ==============================================

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8"
ph = PasswordHasher()

# ==============================================
# Utility Functions (your existing ones)
# ==============================================

def generate_salts() -> Tuple[bytes, bytes]:
    return os.urandom(16), os.urandom(16)

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes):
    # Dummy mapping for demonstration; replace with your real mapping
    plot_type = "cubic"
    p1, p2 = 2.5, 0.75
    # Example way to mix salt randomness
    x = (ord(char) + salt1[0] + salt2[0]) % 255
    return plot_type, p1, p2, x

def generate_user_range(salt1: bytes, salt2: bytes):
    user_x_min = (salt1[0] + 1) * 0.01
    user_x_max = (salt2[0] + 2) * 0.05 + user_x_min
    return user_x_min, user_x_max

def normalize_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max):
    x_min, x_max = min(x_raw_values), max(x_raw_values)
    normalized = [
        user_x_min + (x - x_min) * (user_x_max - user_x_min) / (x_max - x_min + 1e-9)
        for x in x_raw_values
    ]
    return normalized

def apply_plot_function(plot_type: str, p1: float, p2: float, x: float) -> float:
    # Dummy math function for testing — replace with your real one
    if plot_type == "cubic":
        return (x**3 + p1 * x - p2 * x + 1.2356) % 1e6
    else:
        return (math.sin(x) + p1 * x + p2) % 1e6

# ==============================================
# 🔁 Transformation Process
# ==============================================

def transform_input(raw_input: bytes | str, salt1: bytes, salt2: bytes) -> bytes:
    """
    Full mathematical transformation pipeline.
    Can take either string or bytes as input.
    """
    if isinstance(raw_input, str):
        raw_input = raw_input.encode("utf-8")

    combined_input = raw_input + salt1 + salt2
    x_raw_values = []

    for b in combined_input:
        char = chr(b)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        x_raw_values.append(x)

    user_x_min, user_x_max = generate_user_range(salt1, salt2)
    x_values = normalize_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max)

    values = []
    for i, b in enumerate(combined_input):
        char = chr(b)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        result = apply_plot_function(plot_type, p1, p2, x_values[i])
        values.append(result)

    binary_data = struct.pack(f'{len(values)}f', *values)
    return binary_data

# ==============================================
# 🧩 Iterative Transformation Wrapper
# ==============================================

def iterative_transform(password: str, salt1: bytes, salt2: bytes, rounds: int = 3) -> bytes:
    """
    Repeatedly applies the same transformation pipeline.
    Each round's output binary is fed into the next as input.
    """
    current = password
    for i in range(rounds):
        print(f"\n[Iteration {i+1}] Applying transformation...")
        current = transform_input(current, salt1, salt2)
        print(f"[DEBUG] Round {i+1} output (hex preview): {current.hex()[:64]}...")
    return current

# ==============================================
# 🔒 Argon2 Hashing Wrapper
# ==============================================

def hash_with_argon2(password: str, salt1: bytes, salt2: bytes, rounds: int = 3) -> str:
    """
    Runs iterative transformation, then hashes final result with Argon2.
    """
    transformed = iterative_transform(password, salt1, salt2, rounds)
    print("\n[INFO] Final binary length:", len(transformed))
    final_hash = ph.hash(transformed)
    return final_hash

# ==============================================
# 🧠 Example Compare/Benchmark Function
# ==============================================

def compare_methods(password: str, iterations: int = 3):
    """
    Demonstrates multiple hashing methods with your iterative transformation pipeline.
    """
    salt1, salt2 = generate_salts()

    print(f"\n[START] Password: {password}")
    print(f"[INFO] Iterations: {iterations}")

    start_time = time.perf_counter()
    final_hash = hash_with_argon2(password, salt1, salt2, iterations)
    end_time = time.perf_counter()

    print(f"\n[RESULT] Final Argon2 hash after {iterations} transformations:")
    print(final_hash)
    print(f"[TIME] Total time: {end_time - start_time:.4f} seconds")

    # Verification test
    verify_data = iterative_transform(password, salt1, salt2, iterations)
    try:
        ph.verify(final_hash, verify_data)
        print("[VERIFY] ✅ Argon2 verification successful.")
    except Exception as e:
        print("[VERIFY] ❌ Verification failed:", e)

# ==============================================
# 🚀 Main
# ==============================================

if __name__ == "__main__":
    password = "abc"
    iterations = 3  # you can increase this (e.g., 5, 10, etc.)
    compare_methods(password, iterations)
