import os
import hmac
import base64
import getpass
import struct
import math
import numpy as np
from typing import Dict, Tuple
from hashlib import sha256
from argon2 import PasswordHasher
from collections import Counter

# =============================
# Secret key (system-wide constant)
# =============================
SECRET_KEY = b"super_secret_research_key"

# =============================
# Helper functions
# =============================

# def safe_float_from_bytes(b: bytes, precision: int = 6) -> float:
#     """
#     Converts 4 bytes into a safe IEEE-754 float with controlled range and precision.
#     Avoids too many zeros, denormals, NaNs, or extreme values.
#     """
#     # Convert bytes to uint32 bits
#     bits = int.from_bytes(b, byteorder='big')
#     sign = (bits >> 31) & 0x1
#     exponent = (bits >> 23) & 0xFF
#     mantissa = bits & 0x7FFFFF

#     # Clamp exponent to avoid denormals, zeros, Inf, NaN
#     if exponent == 0 or exponent == 255:
#         exponent = 127  # bias for exponent 0 → safe normal number

#     # Clamp mantissa to avoid exact zero
#     if mantissa == 0:
#         mantissa = 1

#     safe_bits = (sign << 31) | (exponent << 23) | mantissa
#     safe_bytes = safe_bits.to_bytes(4, byteorder='big')

#     val = struct.unpack('>f', safe_bytes)[0]

#     # Clamp magnitude to avoid extremes
#     min_val = 1e-6
#     max_val = 1e6
#     val = max(min(val, max_val), -max_val)
#     if abs(val) < min_val:
#         val = min_val if val >= 0 else -min_val

#     return round(val, precision)


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

# def generate_user_param_range(salt1: bytes, salt2: bytes, label: bytes) -> Tuple[float, float]:
#     """
#     Generates a user-specific safe range for parameters (p1, p2).
#     Uses the same strong float extraction as for x normalization,
#     but with different labels for unpredictability.
#     """
#     def extract_strong_float(hmac_bytes: bytes) -> float:
#         val = safe_float_from_bytes(hmac_bytes[:4], precision=6)
#         abs_val = abs(val) if val != 0 else 1.0
#         exponent = math.log10(abs_val)
#         scaled = exponent % 2.0   # clamp to smaller range than x
#         return 10 ** scaled       # in [1.0, 100.0)

#     hmac_min = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_min', sha256).digest()
#     hmac_max = hmac.new(SECRET_KEY, salt1 + salt2 + label + b'_max', sha256).digest()

#     min_val = extract_strong_float(hmac_min)
#     max_val = extract_strong_float(hmac_max)

#     if max_val < min_val:
#         min_val, max_val = max_val, min_val
#     if abs(max_val - min_val) < 1.0:
#         # Add additional gap derived from another HMAC to keep it user-specific
#         separation_hmac = hmac.new(SECRET_KEY, salt1 + salt2 + b'separation', sha256).digest()
#         separation = round(safe_float_from_bytes(separation_hmac[:4]) % 10.0, 6)  # Add up to 10 units
#         max_val = round(min_val + separation + 1.5, 6)  # Final adjusted max value

#     return round(min_val, 6), round(max_val, 6)

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

# =============================
# Plot mapping function
# =============================

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

# =============================
# Secure password hashing pipeline
# =============================


ph = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=4, hash_len=32)

# def secure_hash_password(password: str, salt: bytes) -> str:
#     """Apply plot-based transformation + Argon2 hashing."""
#     transformed_values = []
#     xs = []

#     for ch in password:
#         plot_type, p1, p2, x = map_char_to_function_with_x(ch, salt, salt)
#         xs.append(x)
#         # simple transformation (example: f(x) = p1 * x + p2)
#         y = p1 * x + p2 + plot_type
#         transformed_values.append(round(y, 6))

#     # Normalize x-values into user-specific range
#     xs_norm = normalize_x_values_to_custom_range(xs, 0, 1)

#     # Combine everything into a unique string
#     transformed_str = "|".join(str(v) for v in transformed_values + xs_norm)

#     # Final Argon2 hash
#     # ph = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=4, hash_len=32)
#     return ph.hash(transformed_str)

def secure_hash_password(password: str, salt1: bytes, salt2: bytes) -> bytes:
    combined_input = password.encode() + salt1 + salt2

    x_raw_values = []
    for b in combined_input:
        char = chr(b)
        _, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)
        x_raw_values.append(x)

    user_x_min, user_x_max = generate_user_range(salt1, salt2)
    x_values = normalize_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max)

    values = []
    for i, b in enumerate(combined_input):
        char = chr(b)
        plot_type, p1, p2, _ = map_char_to_function_with_x(char, salt1, salt2)
        values.append(apply_plot_function(plot_type, p1, p2, x_values[i]))

    binary_data = struct.pack(f'{len(values)}f', *values)

    # ph = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=4, hash_len=32)
    return binary_data


# =============================
# User database
# =============================
users: Dict[str, Dict[str, str]] = {}

# def register_user(username: str, password: str):
#     if username in users:
#         print("[!] Username already exists.")
#         return

#     salt = os.urandom(16)
#     hashed_password = secure_hash_password(password, salt)

#     users[username] = {
#         "salt": base64.b64encode(salt).decode(),
#         "hash": hashed_password,
#         "password": password   #  store raw password for debug/demo
#     }
#     print(f"[+] User '{username}' registered successfully!")


# def authenticate_user(username: str, password: str):
#     if username not in users:
#         print("[!] Username not found.")
#         return False

#     user_record = users[username]
#     salt = base64.b64decode(user_record["salt"])
#     stored_hash = user_record["hash"]

#     transformed_values = []
#     xs = []

    # try:
    #     recomputed_hash = secure_hash_password(password, salt)

    #     if recomputed_hash == stored_hash:
    #         print(f"[✓] Authentication successful for '{username}'!")
    #         return True
    #     else:
    #         print("[!] Authentication failed: wrong password.")
    #         return False
    # except Exception as e:
    #     print(f"[!] Authentication error: {e}")
    #     return False

    # for ch in password:
    #     plot_type, p1, p2, x = map_char_to_function_with_x(ch, salt, salt)
    #     xs.append(x)
    #     y = p1 * x + p2 + plot_type
    #     transformed_values.append(round(y, 6))

    # xs_norm = normalize_x_values_to_custom_range(xs, 0, 1)
    # transformed_str = "|".join(str(v) for v in transformed_values + xs_norm)

    # try:
    #     if ph.verify(stored_hash, transformed_str):
    #         print(f"[✓] Authentication successful for '{username}'!")
    #         return True
    # except:
    #     print("[!] Authentication failed: wrong password.")
    #     return False

def register_user(username: str, password: str):
    if username in users:
        print("[!] Username already exists.")
        return

    # Generate two salts for the full pipeline
    salt1 = os.urandom(8)  # 8 bytes for salt1
    salt2 = os.urandom(8)  # 8 bytes for salt2

    transformed = secure_hash_password(password, salt1, salt2)
    hashed_password = ph.hash(transformed)

    users[username] = {
        "salt1": base64.b64encode(salt1).decode(),
        "salt2": base64.b64encode(salt2).decode(),
        "hash": hashed_password,
        "password": password  #  store raw password for debug/demo only
    }
    print(f"[+] User '{username}' registered successfully!")


def authenticate_user(username: str, password: str):
    if username not in users:
        print("[!] Username not found.")
        return False

    user_record = users[username]
    salt1 = base64.b64decode(user_record["salt1"])
    salt2 = base64.b64decode(user_record["salt2"])
    stored_hash = user_record["hash"]

    try:
        transformed = secure_hash_password(password, salt1, salt2)
        print(f"stored hash {stored_hash}, recomputed hash {transformed}")
        if ph.verify(stored_hash, transformed):  # We must verify correctly
            print(f"[✓] Authentication successful for '{username}'!")
            return True
    except Exception as e:
        print(f"[!] Authentication error: {e}")
        return False

    print("[!] Authentication failed: wrong password.")
    return False



# =============================
# CLI loop
# =============================
def main():
    while True:
        print("\n--- User System ---")
        print("1. Register")
        print("2. Login")
        print("3. Show DB (debug)")
        print("4. Exit")

        choice = input("Select an option: ").strip()

        if choice == "1":
            username = input("Enter username: ").strip()
            # 🔹 use normal input so password is visible
            password = input("Enter password: ")  
            register_user(username, password)

        elif choice == "2":
            username = input("Enter username: ").strip()
            password = input("Enter password: ")  # 🔹 visible on typing
            authenticate_user(username, password)

        elif choice == "3":
            print("\n[DEBUG] Current User Database:")
            for user, record in users.items():
                print(f"User: {user}")
                print(f"  Password: {record['password']}")
                print(f"  Salt1: {record['salt1']}")
                print(f"  Salt2: {record['salt2']}")
                print(f"  Hash: {record['hash']}\n")

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("[!] Invalid choice.")


if __name__ == "__main__":
    main()
