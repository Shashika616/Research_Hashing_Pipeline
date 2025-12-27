import os
import hmac
import struct
import hashlib
import bcrypt
import time
import matplotlib.pyplot as plt
import numpy as np
import tracemalloc
from argon2 import PasswordHasher
from hashlib import sha256
from typing import Tuple
from collections import Counter

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

def apply_plot_function(plot_type: int, p1: float, p2: float, x: float = 1.0) -> float:
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
            return p1 * log(p2 * x) if p2 * x > 0 else 0.0
        else:
            return 0.0
    except:
        return 0.0

def map_char_to_function(char: str, salt1: bytes, salt2: bytes) -> Tuple[int, float, float]:
    char_bytes = char.encode('utf-8')
    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'type', sha256).digest()
    plot_type = hmac_type[0] % 6
    hmac_p1 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p1', sha256).digest()
    p1 = struct.unpack('f', hmac_p1[:4])[0]
    hmac_p2 = hmac.new(SECRET_KEY, salt1 + salt2 + char_bytes + b'p2', sha256).digest()
    p2 = struct.unpack('f', hmac_p2[:4])[0]
    return plot_type, p1, p2

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

    # Plot-only
    values = [apply_plot_function(*map_char_to_function(c, salt1, salt2)) for c in password]
    binary_data = struct.pack(f'{len(values)}f', *values)
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
