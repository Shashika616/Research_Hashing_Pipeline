# ============================================
# Full Security Impact Analysis
# Argon2 vs Plot+Argon2
# ============================================

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ----------------------------
# 1. Load Data
# ----------------------------

# file_path = "password_benchmarks.xlsx"  # Change if needed
# df = pd.read_excel(file_path)

# If CSV instead:
df = pd.read_csv("password_benchmarks.csv")

print("File loaded successfully.\n")

# ----------------------------
# 2. Extract Relevant Columns
# ----------------------------

columns_needed = [
    'password',
    'method',
    'entropy',
    'crack_serial_sec',
    'crack_parallel_sec',
    'memory_kb'
]

df = df[columns_needed]

# Clean memory_kb column (remove ' KB' and convert to float)

df['memory_kb'] = (
    df['memory_kb']
    .astype(str)
    .str.replace(" KB", "", regex=False)
)

df['memory_kb'] = pd.to_numeric(df['memory_kb'], errors='coerce')

argon = df[df['method'] == 'Argon2']
plot_argon = df[df['method'] == 'Plot+Argon2']

# ----------------------------
# 3. Merge for Paired Testing
# ----------------------------

merged = pd.merge(
    argon,
    plot_argon,
    on='password',
    suffixes=('_Argon2', '_PlotArgon2')
)

print(f"Paired samples: {len(merged)}\n")

# =============================
# Utility Function
# =============================

def run_wilcoxon(metric_name):
    x = merged[f"{metric_name}_Argon2"]
    y = merged[f"{metric_name}_PlotArgon2"]

    stat, p = wilcoxon(x, y)

    # Effect size calculation
    diff = y - x
    non_zero = diff[diff != 0]
    n = len(non_zero)

    z = (stat - (n*(n+1)/4)) / np.sqrt(n*(n+1)*(2*n+1)/24)
    r = abs(z) / np.sqrt(n)

    median_x = np.median(x)
    median_y = np.median(y)
    median_diff = np.median(diff)

    percent_increase = ((median_y - median_x) / median_x) * 100 if median_x != 0 else 0
    

    print(f"----- {metric_name.upper()} -----")
    print(f"Median Argon2: {median_x:.6f}")
    print(f"Median Plot+Argon2: {median_y:.6f}")
    print(f"Median Difference: {median_diff:.6f}")
    print(f"Relative % Change: {percent_increase:.2f}%")
    print(f"Wilcoxon W: {stat}")
    print(f"P-value: {p:.6f}")
    print(f"Effect Size (r): {r:.4f}")

    if p < 0.05:
        print("Result: Statistically significant difference detected.\n")
    else:
        print("Result: No statistically significant difference detected.\n")

# =============================
# 4. Run Tests
# =============================

print("========== ENTROPY TEST ==========")
run_wilcoxon("entropy")

print("========== SERIAL CRACK TIME TEST ==========")
run_wilcoxon("crack_serial_sec")

print("========== PARALLEL CRACK TIME TEST ==========")
run_wilcoxon("crack_parallel_sec")

print("========== MEMORY USAGE TEST ==========")
run_wilcoxon("memory_kb")