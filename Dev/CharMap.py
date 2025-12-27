import numpy as np
import matplotlib.pyplot as plt

# --- Define plot functions for characters '0' to 'f' ---
def plot_0(x, p1=1.0, p2=1.0):
    y1 = p1 * np.sin(x)
    y2 = p2 * np.cos(np.pi/2 + x)
    return y1, y2

def plot_1(x, p1=1.0, p2=0.0):
    # y = mx + c
    y_line = p1 * x + p2
    # vertical line x = 1
    y_min, y_max = np.min(y_line), np.max(y_line)
    y_vert = np.linspace(y_min - 1, y_max + 1, 100)
    x_vert = np.ones_like(y_vert)
    return y_line, (x_vert, y_vert)

def plot_2(x, p1=1.0, p2=4.0): return -p1 * x**2 + p2
def plot_3(x, p1=1.0, p2=2.0): return p1 * np.sin(x) * x / p2

def plot_4(x, p1=1.0, p2=1.0, base_y=-1.0):
    # y = m * |x - p2|
    # y_line = p1 * np.abs(x - p2)

    # y = mx + c
    y_line = p1 * x + p2
    
    # vertical line at x = 1
    y_min, y_max = np.min(y_line), np.max(y_line)
    y_vert = np.linspace(y_min - 1, y_max + 1, 100)
    x_vert = np.ones_like(y_vert)
    
    # horizontal base line y = base_y
    y_base = np.full_like(x, base_y)
    
    return y_line, (x_vert, y_vert), (x, y_base)

def plot_5(x, p1=1.0, p2=5.0): return p1 * np.clip(np.tan(x), -p2, p2)
def plot_6(x, p1=1.0, p2=3.0): return p1 * (x**3 - p2 * x)
def plot_7(x, p1=1.0, p2=1.0): return p1 * np.floor(x + p2)
def plot_8(x, p1=1.0, p2=2.0): return p1 * np.cos(p2 * x)
def plot_9(x, p1=1.0, p2=1.0): return p1 / (1.0 + np.exp(-p2 * x))
def plot_a(x, p1=1.0, p2=1.0): return p1 * np.sin(p2 * x) * np.exp(-x**2)
def plot_b(x, p1=1.0, p2=0.0): return -p1 * x**2 + p2
def plot_c(x, p1=1.0, p2=1.0): return p1 * np.arctan(p2 * x)
def plot_d(x, p1=1.0, p2=1.0): return p1 * np.log(p2 * (x + 3.1))  # avoid log(0)
def plot_e(x, p1=1.0, p2=1.0): return -p1 * x**3 + p2 * x
def plot_f(x, p1=1.0, p2=0.3): return p1 * np.cos(x) + p2 * x

# Mapping each hex character to its plot function
plot_function_map = {
    '0': plot_0, '1': plot_1, '2': plot_2, '3': plot_3,
    '4': plot_4, '5': plot_5, '6': plot_6, '7': plot_7,
    '8': plot_8, '9': plot_9, 'a': plot_a, 'b': plot_b,
    'c': plot_c, 'd': plot_d, 'e': plot_e, 'f': plot_f
}

# Prepare x-axis range
x_vals = np.linspace(-3, 3, 400)

# Function to draw a group of 4 characters
def draw_plot_group(chars, group_index):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Plot Group {group_index + 1}: Characters {', '.join(chars)}", fontsize=14)
    
    for i, char in enumerate(chars):
        func = plot_function_map[char]
        ax = axes[i]
        try:
            y_vals = func(x_vals)
            
            # For plot_4 triple return (3 parts)
            if isinstance(y_vals, tuple):
                if len(y_vals) == 3:
                    # plot_4 case: line, vertical, horizontal base line
                    y_line = y_vals[0]
                    x_vert, y_vert = y_vals[1]
                    x_base, y_base = y_vals[2]
                    ax.plot(x_vals, y_line, label='y = m|x - p2|', color='blue')
                    ax.plot(x_vert, y_vert, label='x = 1', color='red')
                    ax.plot(x_base, y_base, label=f'y = {y_base[0]:.1f}', color='purple', linestyle='-')
                    ax.legend(fontsize=8)
                elif len(y_vals) == 2:
                    # plot_0 or plot_1 case
                    if isinstance(y_vals[1], tuple) and len(y_vals[1]) == 2:
                        # plot_1 case: line and vertical line
                        y_line = y_vals[0]
                        x_vert, y_vert = y_vals[1]
                        ax.plot(x_vals, y_line, label='y = mx + c', color='blue')
                        ax.plot(x_vert, y_vert, label='x = 1', color='red')
                        ax.legend(fontsize=8)
                    else:
                        # plot_0 case: two y vectors
                        y1, y2 = y_vals
                        ax.plot(x_vals, y1, label='sin(x)', color='blue')
                        ax.plot(x_vals, y2, label='cos(π/2 + x)', color='green')
                        ax.legend(fontsize=8)
            else:
                ax.plot(x_vals, y_vals, color='blue')

            ax.set_title(f"Character '{char}'", fontsize=12)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha='center', va='center', color='red', fontsize=10)
        
        ax.grid(True)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Break characters into 4 groups of 4
char_groups = [
    ['0', '1', '2', '3'],
    ['4', '5', '6', '7'],
    ['8', '9', 'a', 'b'],
    ['c', 'd', 'e', 'f']
]

# Draw each group
for idx, group in enumerate(char_groups):
    draw_plot_group(group, idx)
