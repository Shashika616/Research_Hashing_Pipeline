# ============================================
# Wilcoxon Signed-Rank Test + Visualization
# Argon2 vs Plot+Argon2 (Entropy Comparison)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ----------------------------
# 1. Load Excel File
# ----------------------------

file_path = "password_benchmarks.xlsx"  # Change if needed

# If CSV instead, use:
df = pd.read_csv("password_benchmarks.csv")

# df = pd.read_excel(file_path)

print("File loaded successfully.\n")

# ----------------------------
# 2. Filter Relevant Methods
# ----------------------------

argon = df[df['method'] == 'Argon2'][['password', 'entropy']]
plot_argon = df[df['method'] == 'Plot+Argon2'][['password', 'entropy']]

print(f"Argon2 samples: {len(argon)}")
print(f"Plot+Argon2 samples: {len(plot_argon)}")

# ----------------------------
# 3. Merge for Paired Comparison
# ----------------------------

merged = pd.merge(
    argon,
    plot_argon,
    on='password',
    suffixes=('_Argon2', '_PlotArgon2')
)

print(f"Paired samples: {len(merged)}\n")

# ----------------------------
# 4. Wilcoxon Signed-Rank Test
# ----------------------------

stat, p_value = wilcoxon(
    merged['entropy_Argon2'],
    merged['entropy_PlotArgon2']
)

# ----------------------------
# 5. Effect Size (r)
# ----------------------------

differences = merged['entropy_PlotArgon2'] - merged['entropy_Argon2']
non_zero_diff = differences[differences != 0]
n = len(non_zero_diff)

z = (stat - (n*(n+1)/4)) / np.sqrt(n*(n+1)*(2*n+1)/24)
r = abs(z) / np.sqrt(n)

# ----------------------------
# 6. Print Results
# ----------------------------

print("========== Wilcoxon Signed-Rank Test ==========")
print(f"Test Statistic (W): {stat}")
print(f"P-value: {p_value:.6f}")
print(f"Effect Size (r): {r:.4f}")
print("===============================================\n")

alpha = 0.05

if p_value < alpha:
    print("Result: Statistically significant difference detected.")
else:
    print("Result: No statistically significant difference detected.")

print("\n")

# ----------------------------
# 7. Visualization - Boxplot
# ----------------------------

plt.figure()

plt.boxplot([
    merged['entropy_Argon2'],
    merged['entropy_PlotArgon2']
])

plt.xticks([1, 2], ['Argon2', 'Plot+Argon2'])
plt.ylabel("Entropy")
plt.title("Entropy Distribution: Argon2 vs Plot+Argon2")

plt.show()


# ----------------------------
# 8. Visualization - Paired Difference Plot
# ----------------------------

plt.figure()

for i in range(len(merged)):
    plt.plot(
        ['Argon2', 'Plot+Argon2'],
        [merged['entropy_Argon2'].iloc[i],
         merged['entropy_PlotArgon2'].iloc[i]]
    )

plt.ylabel("Entropy")
plt.title("Paired Entropy Differences")

plt.show()