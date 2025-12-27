import os
import hmac
import base64
import hashlib
import struct
import math
import numpy as np
from typing import Dict, Tuple
from hashlib import sha256
from typing import Tuple, List
from argon2 import PasswordHasher
from collections import Counter

SECRET_KEY = b"super_secret_research_key"


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

def apply_plot_function(plot_type: int, p1: float, p2: float, x_list: List[float], ) -> Tuple[float, str]:
    """
    Extended apply_plot_function with many chaotic / nonlinear plot types.
    Returns (expr_val, func_repr).
    Supported plot_type:
      0..5  -> original functions (unchanged)
      6..50 -> additional maximum-security chaotic functions
    """
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
            xi = x_list[i]
            term_val = p1 * xi / (1 + xi**2) - p2 * xi**3
            expr_val += term_val
            expr_str_parts.append(f"({p1}*x{i+1}/(1+x{i+1}^2) - {p2}*x{i+1}^3)")

    elif plot_type == 5:
        for i in range(n):
            for j in range(n):
                term_val = ((-1)**i)*p1*math.sin(x_list[i]) + p2*math.exp(x_list[j]) - math.log(1 + abs(x_list[i]*x_list[j]))
                expr_val += term_val
                expr_str_parts.append(f"({(-1)**i}*{p1}*sin(x{i+1}) + {p2}*exp(x{j+1}) - log(1+|x{i+1}*x{j+1}|))")

    # --- additional chaotic / max-security functions (6..50) ---
    elif plot_type == 6:
        # High-order mixed polynomial with alternating signs
        for i in range(n):
            for j in range(i, n):
                term = ((-1)**(i+j)) * p1 * (x_list[i]**2) * x_list[j] - p2 * (x_list[j]**4)
                expr_val += term
                expr_str_parts.append(f"({(-1)**(i+j)}*{p1}*x{i+1}^2*x{j+1} - {p2}*x{j+1}^4)")

    elif plot_type == 7:
        # Logistic-like sensitivity + cross-coupling
        for i in range(n):
            xi = x_list[i]
            logistic = xi * (1 - xi) if -1 < xi < 2 else math.tanh(xi)  # fallback to tanh outside safe range
            expr_val += p1 * logistic
            expr_str_parts.append(f"({p1}*logistic(x{i+1}))")
        # cross terms
        for i in range(n-1):
            expr_val += p2 * x_list[i] * x_list[i+1]
            expr_str_parts.append(f"({p2}*x{i+1}*x{i+2})")

    elif plot_type == 8:
        # Henon-like 2D coupling expanded across vector
        for i in range(0, n, 2):
            x0 = x_list[i]
            x1 = x_list[i+1] if i+1 < n else 0.0
            henon = 1 - p1 * x0**2 + p2 * x1
            expr_val += henon
            expr_str_parts.append(f"(henon({i+1},{i+2}))")

    elif plot_type == 9:
        # Nested trig with product mixing
        prod = 1.0
        for i in range(n):
            prod *= math.sin(x_list[i]) + 1.0
            expr_str_parts.append(f"sin(x{i+1})")
        expr_val += p1 * prod - p2 * sum(x**2 for x in x_list)

    elif plot_type == 10:
        # Reciprocal-sensitive and sign flips
        for i in range(n):
            xi = x_list[i]
            term = p1 * math.copysign(1.0, xi) / (1 + abs(xi)) - p2 * (xi**3) / (1 + abs(xi))
            expr_val += term
            expr_str_parts.append(f"(sign(x{i+1})/(1+|x{i+1}|) - {p2}*x{i+1}^3/(1+|x{i+1}|))")

    elif plot_type == 11:
        # High-frequency sine cascade
        for i in range(n):
            xi = x_list[i]
            term = p1 * math.sin(xi * (i+1) * 7.13) + p2 * math.sin(math.sin(xi) * 13.37)
            expr_val += term
            expr_str_parts.append(f"({p1}*sin({i+1}*7.13*x{i+1}) + {p2}*sin(sin(x{i+1})*13.37))")

    elif plot_type == 12:
        # tanh-sinh squashing plus polynomial
        for i in range(n):
            xi = x_list[i]
            term = p1 * math.tanh(xi) + p2 * (math.sinh(xi) / (1 + xi**2))
            expr_val += term
            expr_str_parts.append(f"({p1}*tanh(x{i+1}) + {p2}*sinh(x{i+1})/(1+x{i+1}^2))")

    elif plot_type == 13:
        # Nested log + product-of-abs
        prod_abs = 1.0
        for i in range(n):
            prod_abs *= (1 + abs(x_list[i]))
            expr_str_parts.append(f"(1+|x{i+1}|)")
        expr_val += p1 * math.log1p(prod_abs) - p2 * sum(math.log1p(1+abs(x)) for x in x_list)

    elif plot_type == 14:
        # Fractional exponents + alternating sums
        for i in range(n):
            xi = x_list[i]
            term = p1 * (abs(xi) ** 0.5) * ((-1)**i) + p2 * (xi**5) / (1 + abs(xi))
            expr_val += term
            expr_str_parts.append(f"({p1}*|x{i+1}|^0.5*{(-1)**i} + {p2}*x{i+1}^5/(1+|x{i+1}|))")

    elif plot_type == 15:
        # Rational interaction network
        for i in range(n):
            for j in range(n):
                denom = 1 + abs(x_list[i] * x_list[j]) + (i+1)
                term = p1 * (x_list[i] - x_list[j]) / denom
                expr_val += term
                expr_str_parts.append(f"({p1}*(x{i+1}-x{j+1})/(1+|x{i+1}*x{j+1}|+{i+1}))")

    elif plot_type == 16:
        # Chaotic map chain: iterated logistic-like on xi
        for i in range(n):
            xi = x_list[i]
            y = xi
            for k in range(3):  # iterate few times
                y = p1 * y * (1 - y) if -1 < y < 2 else math.tanh(p1*y)
            expr_val += y + p2 * xi**2
            expr_str_parts.append(f"(iter_logistic_{i+1})")

    elif plot_type == 17:
        # Symmetric polynomial with sine modulation
        for i in range(n):
            s = sum(x_list)
            term = p1 * (x_list[i]**2) * math.sin(s) - p2 * x_list[i]
            expr_val += term
            expr_str_parts.append(f"({p1}*x{i+1}^2*sin(sum(x)) - {p2}*x{i+1})")

    elif plot_type == 18:
        # Mix of atan and reciprocal damping
        for i in range(n):
            xi = x_list[i]
            term = p1 * math.atan(xi) - p2 * xi / (1 + abs(xi))
            expr_val += term
            expr_str_parts.append(f"({p1}*atan(x{i+1}) - {p2}*x{i+1}/(1+|x{i+1}|))")

    elif plot_type == 19:
        # Exponential damping of higher powers
        for i in range(n):
            xi = x_list[i]
            term = p1 * math.exp(-abs(xi)) * xi**4 - p2 * math.log1p(abs(xi))
            expr_val += term
            expr_str_parts.append(f"({p1}*exp(-|x{i+1}|)*x{i+1}^4 - {p2}*log1p(|x{i+1}|))")

    elif plot_type == 20:
        # Cross-sine products and geometric decay
        for i in range(n):
            for j in range(i+1, n):
                term = p1 * math.sin(x_list[i]) * math.sin(x_list[j]) / (1 + abs(x_list[i]-x_list[j]))
                expr_val += term
                expr_str_parts.append(f"({p1}*sin(x{i+1})*sin(x{j+1})/(1+|x{i+1}-x{j+1}|))")
        expr_val -= p2 * sum(x**2 for x in x_list)

    elif plot_type == 21:
        # Piecewise sign-dependent cubic
        for i in range(n):
            xi = x_list[i]
            if xi >= 0:
                term = p1 * xi**3
            else:
                term = -p2 * abs(xi)**1.5
            expr_val += term
            expr_str_parts.append(f"(piecewise_cubic_{i+1})")

    elif plot_type == 22:
        # Tanh-of-product network
        for i in range(n):
            prod = 1.0
            for j in range(n):
                prod *= (1 + 0.1 * x_list[j])
            expr_val += p1 * math.tanh(prod) - p2 * x_list[i]
            expr_str_parts.append(f"(tanh_prod_all * x{i+1})")

    elif plot_type == 23:
        # High-order cross-power sums
        for i in range(n):
            for j in range(n):
                term = p1 * (x_list[i]**3) * (x_list[j]**2) - p2 * (x_list[i] * x_list[j])
                expr_val += term
                expr_str_parts.append(f"({p1}*x{i+1}^3*x{j+1}^2 - {p2}*x{i+1}*x{j+1})")

    elif plot_type == 24:
        # Nested logs and sines
        for i in range(n):
            xi = x_list[i]
            term = p1 * math.sin(math.log1p(1 + abs(xi))) + p2 * math.log1p(abs(math.sin(xi)))
            expr_val += term
            expr_str_parts.append(f"(sin(log1p(1+|x{i+1}|)) + {p2}*log1p(|sin(x{i+1})|))")

    elif plot_type == 25:
        # Alternating product-sum with index weights
        for i in range(n):
            weight = (i+1)
            term = p1 * weight * x_list[i] * (sum(x_list) - x_list[i]) - p2 * (x_list[i]**2)
            expr_val += term
            expr_str_parts.append(f"({p1}*{weight}*x{i+1}*(sum-x{i+1}) - {p2}*x{i+1}^2)")

    elif plot_type == 26:
        # Exponential-of-sine grid mixing
        for i in range(n):
            for j in range(n):
                term = p1 * math.exp(math.sin(x_list[i] * x_list[j])) - p2 * math.cos(x_list[i] + x_list[j])
                expr_val += term
                expr_str_parts.append(f"(exp(sin(x{i+1}*x{j+1})) - {p2}*cos(x{i+1}+x{j+1}))")

    elif plot_type == 27:
        # Product of shifted reciprocals
        for i in range(n):
            term = p1
            for j in range(n):
                term *= 1.0 / (1 + (x_list[j]**2) + 0.1*(i+1))
            expr_val += term - p2 * sum(x_list)
            expr_str_parts.append(f"(prod_reciprocal_{i+1})")

    elif plot_type == 28:
        # Chaotic envelope: sum of random-seeming deterministic oscillations
        for i in range(n):
            xi = x_list[i]
            term = p1 * (math.sin(xi) * math.cos(xi*1.618) + math.sin(xi*2.718)) - p2 * xi
            expr_val += term
            expr_str_parts.append(f"(chaotic_osc_{i+1})")

    elif plot_type == 29:
        # High-degree odd polynomials with alternating sign
        for i in range(n):
            xi = x_list[i]
            term = p1 * (xi**7) * ((-1)**i) + p2 * (xi**3)
            expr_val += term
            expr_str_parts.append(f"({p1}*x{i+1}^7*{(-1)**i} + {p2}*x{i+1}^3)")

    elif plot_type == 30:
        # Cross-coupled exponential/log ratio terms
        for i in range(n):
            for j in range(n):
                denom = 1 + abs(x_list[i]) + abs(x_list[j])
                term = p1 * math.exp(x_list[i]) / denom - p2 * math.log1p(abs(x_list[j])) / (1+denom)
                expr_val += term
                expr_str_parts.append(f"(exp(x{i+1})/denom - {p2}*log1p(|x{j+1}|)/(...))")

    elif plot_type == 31:
        # Nested chaotic map using atan and cubic
        for i in range(n):
            xi = x_list[i]
            y = math.atan(p1 * xi) ** 3
            expr_val += y - p2 * xi
            expr_str_parts.append(f"(atan({p1}*x{i+1})^3 - {p2}*x{i+1})")

    elif plot_type == 32:
        # Sine-of-products plus scaled sum of cubes
        for i in range(n):
            for j in range(n):
                expr_val += p1 * math.sin(x_list[i] * x_list[j]) + p2 * (x_list[i]**3 + x_list[j]**3)
                expr_str_parts.append(f"(sin(x{i+1}*x{j+1}) + {p2}*(x{i+1}^3+x{j+1}^3))")

    elif plot_type == 33:
        # Asymmetric damping with hyperbolic functions
        for i in range(n):
            xi = x_list[i]
            term = p1 * (math.sinh(xi) / (1 + xi**2)) - p2 * (math.cosh(xi) / (1 + abs(xi)))
            expr_val += term
            expr_str_parts.append(f"(sinh(x{i+1})/(1+x{i+1}^2) - {p2}*cosh(x{i+1})/(1+|x{i+1}|))")

    elif plot_type == 34:
        # Cross-indexed cubic-sqrt interactions
        for i in range(n):
            for j in range(n):
                term = p1 * (x_list[i]**3) * math.sqrt(1 + abs(x_list[j])) - p2 * (x_list[j]**2)
                expr_val += term
                expr_str_parts.append(f"(x{i+1}^3*sqrt(1+|x{j+1}|) - {p2}*x{j+1}^2)")

    elif plot_type == 35:
        # Log-modulated polynomial ring
        for i in range(n):
            xi = x_list[i]
            term = p1 * xi * (math.log1p(1+abs(sum(x_list))) ) - p2 * (xi**2)
            expr_val += term
            expr_str_parts.append(f"(x{i+1}*log1p(1+sum)| - {p2}*x{i+1}^2)")

    elif plot_type == 36:
        # Alternating exponent-sin network
        for i in range(n):
            for j in range(n):
                term = ((-1)**(i+j)) * p1 * math.exp(math.sin(x_list[i] + x_list[j])) - p2 * (x_list[i] - x_list[j])**2
                expr_val += term
                expr_str_parts.append(f"(((-1)^{i+j})*exp(sin(x{i+1}+x{j+1})) - {p2}*(x{i+1}-x{j+1})^2)")

    elif plot_type == 37:
        # Recursive-like accumulation (bounded small loops)
        acc = 0.0
        for i in range(n):
            local = x_list[i]
            for k in range(2):
                local = math.sin(local) * p1 - p2 * (local**2)
            acc += local
            expr_str_parts.append(f"(rec_acc_{i+1})")
        expr_val += acc

    elif plot_type == 38:
        # Complex rational-exponential mixture
        for i in range(n):
            xi = x_list[i]
            term = p1 * (math.exp(xi) / (1 + xi**2)) - p2 * (1 / (1 + math.exp(-abs(xi))))
            expr_val += term
            expr_str_parts.append(f"(exp(x{i+1})/(1+x{i+1}^2) - {p2}/(1+exp(-|x{i+1}|)))")

    elif plot_type == 39:
        # Mixed-order interactions with index weighting and sign flips
        for i in range(n):
            for j in range(n):
                term = (i+1) * p1 * (x_list[i]**2) * math.sin(x_list[j]) - (j+1) * p2 * math.cos(x_list[i] * x_list[j])
                expr_val += term
                expr_str_parts.append(f"({i+1}*{p1}*x{i+1}^2*sin(x{j+1}) - {j+1}*{p2}*cos(x{i+1}*x{j+1}))")

    elif plot_type == 40:
        # Tent-map-like absolute value mixing + cross power
        for i in range(n):
            xi = x_list[i]
            tent = 1 - 2 * abs((xi % 1.0) - 0.5)  # deterministic tent-like map
            expr_val += p1 * tent + p2 * (xi**2)
            expr_str_parts.append(f"(tent(x{i+1})*{p1} + {p2}*x{i+1}^2)")

    elif plot_type == 41:
        # Interaction graph Laplacian-like sum
        total = 0.0
        mean = sum(x_list) / n if n>0 else 0.0
        for i in range(n):
            total += p1 * (x_list[i] - mean)**2
            expr_str_parts.append(f"((x{i+1}-mean)^2)")
        expr_val += total - p2 * mean

    elif plot_type == 42:
        # High-frequency modulated exponential
        for i in range(n):
            xi = x_list[i]
            term = p1 * math.exp(math.sin(xi*17.3)) - p2 * math.sin(xi*3.31)
            expr_val += term
            expr_str_parts.append(f"(exp(sin(17.3*x{i+1})) - {p2}*sin(3.31*x{i+1}))")

    elif plot_type == 43:
        # Product-sum chaotic coupling with index shift
        for i in range(n):
            prod = 1.0
            for j in range(n):
                prod *= (1 + 0.01 * (x_list[(j+i) % n]))
            expr_val += p1 * prod - p2 * sum(x_list)
            expr_str_parts.append(f"(prod_shift_{i+1})")

    elif plot_type == 44:
        # Sine-cube + logarithmic damping
        for i in range(n):
            xi = x_list[i]
            term = p1 * (math.sin(xi)**3) - p2 * math.log1p(1+abs(xi))
            expr_val += term
            expr_str_parts.append(f"(sin(x{i+1})^3 - {p2}*log1p(1+|x{i+1}|))")

    elif plot_type == 45:
        # Mixed fractional power network
        for i in range(n):
            for j in range(n):
                term = p1 * (abs(x_list[i])**0.7) * (abs(x_list[j])**0.3) - p2 * (x_list[i] * x_list[j])
                expr_val += term
                expr_str_parts.append(f"(abs(x{i+1})^0.7*abs(x{j+1})^0.3 - {p2}*x{i+1}*x{j+1})")

    elif plot_type == 46:
        # Exponential-of-product minus damped log-sum
        prod = 1.0
        for x in x_list:
            prod *= (1 + 0.001 * x)
        expr_val += p1 * math.exp(prod) - p2 * math.log1p(1 + abs(sum(x_list)))
        expr_str_parts.append("exp(prod_small_x) - p2*log1p(1+|sum|)")

    elif plot_type == 47:
        # Sinusoidal ring with index-phase
        for i in range(n):
            phase = (i+1) * 0.314159
            expr_val += p1 * math.sin(x_list[i] + phase) - p2 * (x_list[i]**2)
            expr_str_parts.append(f"(sin(x{i+1}+{phase}) - {p2}*x{i+1}^2)")

    elif plot_type == 48:
        # Cross-correlation style chaotic score
        for i in range(n):
            for j in range(i+1, n):
                corr = x_list[i] * x_list[j] / (1 + abs(x_list[i]) + abs(x_list[j]))
                expr_val += p1 * corr - p2 * (x_list[i] - x_list[j])**2
                expr_str_parts.append(f"(corr_{i+1}_{j+1})")

    elif plot_type == 49:
        # Index-weighted nested trig-exp network
        for i in range(n):
            xi = x_list[i]
            inner = math.sin(xi) * math.cos(xi*0.7) + math.tanh(xi*0.1)
            expr_val += (i+1) * p1 * math.exp(inner) - p2 * xi
            expr_str_parts.append(f"((i+1)*exp(inner_{i+1}) - {p2}*x{i+1})")

    elif plot_type == 50:
        # Final aggressive mix: sums, products, logs, exponentials, and sign flips
        s = sum(x_list)
        prod = 1.0
        for i, xi in enumerate(x_list):
            prod *= (1 + 0.001*xi)
            expr_val += ((-1)**i) * p1 * math.sin(xi) * math.exp(-abs(xi)/10.0)
            expr_str_parts.append(f"((-1)^{i}*{p1}*sin(x{i+1})*exp(-|x{i+1}|/10))")
        expr_val += p2 * math.log1p(1 + abs(s)) - 0.5 * prod

    else:
        # Fallback: simple safe linear mapping if unknown plot_type
        for i in range(n):
            expr_val += p1 * x_list[i] - p2 * (x_list[i]**2)
            expr_str_parts.append(f"({p1}*x{i+1} - {p2}*x{i+1}^2)")

    func_repr = " + ".join(expr_str_parts)
    return expr_val, func_repr

def map_char_to_function_with_x(char: str, salt1: bytes, salt2: bytes, char_index: int, max_dims: int = 8) -> Tuple[int, float, float, int, List[float]]:

    # plot type
    tag = char.encode('utf-8') + bytes([char_index])  # use i as unique per character instance
    hmac_type = hmac.new(SECRET_KEY, salt1 + salt2 + tag + b'type', sha256).digest()
    plot_type = hmac_type[0] % 51
    # print("  plot_type -> {}", plot_type)
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

    # print("raw_p1 before normalization:", raw_p1)
    # print("p1 after normalization:", p1)
    # print("raw_p2 before normalization:", raw_p2)
    # print("p2 after normalization:", p2)

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

    # print("  final mapping: plot_type={}, p1={}, p2={}, n={}, x_list={}", plot_type, p1, p2, n, x_list)
    return plot_type, p1, p2, n, x_list

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

def iterative_plot_transform(password: str, salt1: bytes, salt2: bytes, iterations: int):
 
    current_input = password.encode()
    all_intermediates = []

    for it in range(1, iterations + 1):
        combined_input = current_input + salt1 + salt2

        x_raw_values = []  
        char_maps = []

        # Map each byte/char to function and gather raw x vectors
        for idx, b in enumerate(combined_input):
            char = chr(b % 256)
            mapping = map_char_to_function_with_x(
                char, salt1, salt2,
                char_index=idx,
            )
            char_maps.append(mapping)
            x_raw_values.append(mapping[4]) # x_list

        # Normalize per-char vectors into the user-specific range
        user_x_min, user_x_max = generate_user_range(salt1, salt2)
        x_values_multi = normalize_multi_dim_x_values_to_custom_range(x_raw_values, user_x_min, user_x_max, "log")

         # Show normalized multi-dim vectors
        # for idx, vec in enumerate(x_values_multi):
        #     print("    normalized vector idx={} -> {}", idx, vec)

        #Re-normalize into a safe mathematical range for log and exp
        x_values_multi = normalize_multi_dim_x_values_to_custom_range(x_values_multi, -700, 700, "log")

        # # Show normalized multi-dim vectors
        # for idx, vec in enumerate(x_values_multi):
        #     print("    Math normalized vector idx={} -> {}", idx, vec)


        # Apply plot functions (one scalar result per character)
        values = []
        for i, (plot_type, p1, p2, n, _) in enumerate(char_maps):
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
        current_input = hashlib.sha256(binary_data).digest()

    return current_input, all_intermediates




ph = PasswordHasher(time_cost=3, memory_cost=64 * 1024, parallelism=4, hash_len=32)


def secure_hash_password(password: str, salt1: bytes, salt2: bytes) -> bytes:
    # combined_input = password.encode() + salt1 + salt2

    iterations = 10  

    final_binary, all_intermediates = iterative_plot_transform(password, salt1, salt2, iterations)

    return final_binary

# User database

users: Dict[str, Dict[str, str]] = {}

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
        print(f"\nStored hash {stored_hash}, \n\nRecomputed hash {transformed.hex()}")
        if ph.verify(stored_hash, transformed):  
            print(f"[✓] Authentication successful for '{username}'!")
            return True
    except Exception as e:
        print(f"[!] Authentication error: {e}")
        return False

    print("[!] Authentication failed: wrong password.")
    return False

# CLI loop

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
