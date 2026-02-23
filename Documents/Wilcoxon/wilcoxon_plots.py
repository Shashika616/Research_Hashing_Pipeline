# ============================================
# Full Security Impact Analysis + Visualization (Display Inline)
# Argon2 vs Plot+Argon2
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ----------------------------
# 1. Load Data
# ----------------------------

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
df['memory_kb'] = df['memory_kb'].astype(str).str.replace(" KB", "", regex=False)
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
# 4. Utility Function
# =============================

def run_wilcoxon_and_plot(metric_name, y_label):
    x = merged[f"{metric_name}_Argon2"]
    y = merged[f"{metric_name}_PlotArgon2"]

    # --- Wilcoxon test ---
    stat, p = wilcoxon(x, y)
    diff = y - x
    non_zero = diff[diff != 0]
    n = len(non_zero)
    z = (stat - (n*(n+1)/4)) / np.sqrt(n*(n+1)*(2*n+1)/24)
    r = abs(z) / np.sqrt(n)
    median_x = np.median(x)
    median_y = np.median(y)
    median_diff = np.median(diff)
    percent_increase = ((median_y - median_x) / median_x) * 100 if median_x != 0 else 0

    # --- Print results ---
    print(f"----- {metric_name.upper()} -----")
    print(f"Median Argon2: {median_x:.6f}")
    print(f"Median Plot+Argon2: {median_y:.6f}")
    print(f"Median Difference: {median_diff:.6f}")
    print(f"Relative % Change: {percent_increase:.2f}%")
    print(f"Wilcoxon W: {stat}")
    print(f"P-value: {p:.6f}")
    print(f"Effect Size (r): {r:.4f}")
    print("Result: " + ("Statistically significant difference detected.\n" if p < 0.05 else "No statistically significant difference detected.\n"))

    # --- Boxplot ---
    plt.figure(figsize=(6,4))
    plt.boxplot([x, y], labels=['Argon2', 'Plot+Argon2'])
    plt.ylabel(y_label)
    plt.title(f"{metric_name.capitalize()} Boxplot")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # --- Paired difference plot ---
    plt.figure(figsize=(6,4))
    for i in range(len(merged)):
        plt.plot(
            ['Argon2', 'Plot+Argon2'],
            [x.iloc[i], y.iloc[i]],
            marker='o', color='gray', alpha=0.5
        )
    plt.ylabel(y_label)
    plt.title(f"{metric_name.capitalize()} Paired Differences")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# =============================
# 5. Run Tests and Display Figures
# =============================

metrics = [
    ("entropy", "Entropy"),
    ("crack_serial_sec", "Serial Crack Time (s)"),
    ("crack_parallel_sec", "Parallel Crack Time (s)"),
    ("memory_kb", "Memory Usage (KB)")
]

for metric, label in metrics:
    run_wilcoxon_and_plot(metric, label)