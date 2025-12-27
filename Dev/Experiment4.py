import os
import hmac
import struct
import hashlib
import math
import numpy as np
import matplotlib.pyplot as plt

SECRET_KEY = b"5a3f47e9a4d3c8b7e1f5d6c3b8a1f7e3456f78d3b9e1f4a6c8d2e4b6a1f3c7e2"

# ----------------------------
# Utilities
# ----------------------------
def safe_float_from_bytes(b: bytes, precision: int = 6) -> float:
    bits = int.from_bytes(b, byteorder='big')
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
    min_val, max_val = 1e-6, 1e6
    val = max(min(val, max_val), -max_val)
    if abs(val) < min_val:
        val = min_val if val >= 0 else -min_val
    return round(val, precision)

# ----------------------------
# Plot function
# ----------------------------
def apply_plot_function(plot_type: int, p1: float, p2: float, x: float) -> float:
    MAX_MAG = 1e12
    MIN_MAG = 1e-12
    def clamp(v):
        if math.isnan(v) or math.isinf(v):
            return 0.0
        if v > MAX_MAG: return MAX_MAG
        if v < -MAX_MAG: return -MAX_MAG
        if abs(v) < MIN_MAG: return MIN_MAG * (1 if v >= 0 else -1)
        return v
    try:
        if plot_type == 0:
            val = p1*x**8 - p2*x**6 + p1**2*x**5 + math.sin(p1*x**3) + math.exp(p2*x/(1+abs(x))) - p1*math.log(abs(x)+10)
            eqn = f"y = {p1:.2f}x⁸ - {p2:.2f}x⁶ + {p1**2:.2f}x⁵ + sin({p1:.2f}x³) + exp({p2:.2f}x/(1+|x|)) - {p1:.2f}log(|x|+10)"
        elif plot_type == 1:
            val = ((p1*x**12 - p2*x**9 + p1*x**4)/(1+math.exp(-p2*x)) + math.tan(math.sin(p1*x)))
            eqn = f"y = (({p1:.2f}x¹² - {p2:.2f}x⁹ + {p1:.2f}x⁴)/(1+exp(-{p2:.2f}x)) + tan(sin({p1:.2f}x)))"
        elif plot_type == 2:
            val = p1*x**10 + p2*x**7 - p1*x**3 + math.sin(x**2) + math.log(abs(p2*x)+5) - math.exp(-abs(x))
            eqn = f"y = {p1:.2f}x¹⁰ + {p2:.2f}x⁷ - {p1:.2f}x³ + sin(x²) + log(|{p2:.2f}x|+5) - exp(-|x|)"
        elif plot_type == 3:
            val = p1*math.exp(math.sin(p2*x)) - p2*math.log(abs(x*p1)+2) + x**5 - x**4 + p1*x**2
            eqn = f"y = {p1:.2f}exp(sin({p2:.2f}x)) - {p2:.2f}log(|x*{p1:.2f}|+2) + x⁵ - x⁴ + {p1:.2f}x²"
        elif plot_type == 4:
            val = math.sin(p1*x**3) + math.cos(p2*x**2) + math.tan(p1*x/(1+abs(x)))
            eqn = f"y = sin({p1:.2f}x³) + cos({p2:.2f}x²) + tan({p1:.2f}x/(1+|x|))"
        elif plot_type == 5:
            val = p1*x**20 - p2*x**17 + p1*x**14 - p1*p2*x**10 + p2*x**8 + math.sin(x) + math.exp(-abs(p1*x))
            eqn = f"y = {p1:.2f}x²⁰ - {p2:.2f}x¹⁷ + {p1:.2f}x¹⁴ - {p1*p2:.2f}x¹⁰ + {p2:.2f}x⁸ + sin(x) + exp(-|{p1:.2f}x|)"
        else:
            val = 0.0
            eqn = "y = 0"
        return clamp(val), eqn
    except:
        return 0.0, "y = error"

# ----------------------------
# Map character to function & params
# ----------------------------
def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes, user_range=(0.1,1.0), param_range=(0.5,2.0)):
    char_bytes = char.encode('utf-8')
    hmac_type = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'type', hashlib.sha256).digest()
    plot_type = hmac_type[0] % 6

    hmac_p1 = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'p1', hashlib.sha256).digest()
    p1_raw = safe_float_from_bytes(hmac_p1[:4])
    p1 = param_range[0] + (param_range[1]-param_range[0]) * abs(p1_raw % 1)

    hmac_p2 = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'p2', hashlib.sha256).digest()
    p2_raw = safe_float_from_bytes(hmac_p2[:4])
    p2 = param_range[0] + (param_range[1]-param_range[0]) * abs(p2_raw % 1)

    hmac_x = hmac.new(SECRET_KEY, salt1+salt2+char_bytes+b'x', hashlib.sha256).digest()
    x_raw = safe_float_from_bytes(hmac_x[:4])
    x = user_range[0] + (user_range[1]-user_range[0]) * abs(x_raw % 1)

    return plot_type, p1, p2, x

# ----------------------------
# Plot side-by-side graphs for each character with equation
# ----------------------------
def plot_character_graphs_side_by_side(password: str, salt1: bytes, salt2: bytes, delta=1e-7, full_range=5.0):
    combined_input = password.encode() + salt1 + salt2

    for idx, b in enumerate(combined_input):
        char = chr(b % 256)
        plot_type, p1, p2, x = map_char_to_function_with_x(char, salt1, salt2)

        # --- Full function graph ---
        x_full = np.linspace(x - full_range, x + full_range, 1000)
        y_full = np.array([apply_plot_function(plot_type, p1, p2, xv)[0] for xv in x_full])
        eqn = apply_plot_function(plot_type, p1, p2, x)[1]

        # --- Δy graph ---
        dx = np.linspace(-delta*50, delta*50, 200)
        y_base = apply_plot_function(plot_type, p1, p2, x)[0]
        y_dx = np.array([apply_plot_function(plot_type, p1, p2, x + d)[0] - y_base for d in dx])

        # --- Side-by-side subplots ---
        plt.figure(figsize=(14,5))

        plt.subplot(1,2,1)
        plt.plot(x_full, y_full, label='y(x)')
        plt.title(f"Char '{char}' — Full function\n{eqn}", fontsize=10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(dx, y_dx, label='Δy vs Δx', color='orange')
        plt.title(f"Char '{char}' — Δy", fontsize=10)
        plt.xlabel("Δx")
        plt.ylabel("Δy")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    password = "abc"
    salt1 = os.urandom(8)
    salt2 = os.urandom(8)

    plot_character_graphs_side_by_side(password, salt1, salt2)
