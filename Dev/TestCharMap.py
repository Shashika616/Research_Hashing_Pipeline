import numpy as np
import matplotlib.pyplot as plt

# --- Define plot functions for characters '0' to 'f' ---

def plot_0(x_dummy, p1=1.0, p2=1.0):
    # Oval using parametric ellipse equation
    t = np.linspace(0, 2 * np.pi, 400)
    x = p1 * np.cos(t)
    y = p2 * np.sin(t)
    return x, y

def plot_1(x, p1=1.0, p2=0.0):
    y_line = p1 * x + p2
    y_min, y_max = np.min(y_line), np.max(y_line)
    y_vert = np.linspace(y_min - 1, y_max + 1, 100)
    x_vert = np.ones_like(y_vert)
    return y_line, (x_vert, y_vert)

def plot_2(x_dummy, p1=10.0, p2=4.0, base_y=-2.0):
    y_vals = np.linspace(-2, 2, 400)
    x_parabola = -p1 * y_vals**2 + p2
    x_base = np.linspace(-50, 50, 400)
    y_base = np.full_like(x_base, base_y)
    return (x_parabola, y_vals), (x_base, y_base)

def plot_3(x_dummy, p1=1.0, p2=2.0):
    # Rotated function to resemble number "3"
    y_vals = np.linspace(-3, 3, 400)
    x_vals = p1 * np.sin(y_vals) * y_vals / p2
    return x_vals, y_vals

def plot_4(x, p1=1.0, p2=1.0, base_y=-1.0):
    y_line = p1 * x + p2
    y_min, y_max = np.min(y_line), np.max(y_line)
    y_vert = np.linspace(y_min - 1, y_max + 1, 100)
    x_vert = np.ones_like(y_vert)
    y_base = np.full_like(x, base_y)
    return y_line, (x_vert, y_vert), (x, y_base)

def plot_5(x_dummy=None):
    # Top horizontal line (y = 2) from x = 2 to 5
    x_top = np.linspace(2, 5, 100)
    y_top = np.full_like(x_top, 2)

    # Middle vertical line (x = 2) from y = 0.5 to 2
    y_vert = np.linspace(0.5, 2, 100)
    x_vert = np.full_like(y_vert, 2)

    # Middle horizontal line (y = 0.5) from x = 2 to 4.5
    x_mid = np.linspace(2, 5, 100)
    y_mid = np.full_like(x_mid, 0.5)

    # Bottom curve (parabola): x = -0.8y^2 + 5, from y = -2 to 0.5
    y_curve = np.linspace(-2, 0.5, 200)
    x_curve = -0.8 * y_curve**2 + 5

    return (x_top, y_top), (x_vert, y_vert), (x_mid, y_mid), (x_curve, y_curve)


def plot_6(x_dummy=None, radius=2.0, center_x=0.0, center_y=0.0):
    # 1. Left half-circle (loop) from bottom to top (π/2 to 3π/2)
    theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 200)
    x_circle = center_x + radius * np.cos(theta)
    y_circle = center_y + radius * np.sin(theta)

    # 2. Vertical line from bottom to center (|), aligned with leftmost point of circle
    x_vert = np.full(100, center_x )  # midle edge of the circle
    y_vert = np.linspace(center_y - radius, center_y, 100)

    # 3. Horizontal line (-) from top of vertical to right (into the circle)
    x_horz = np.linspace(center_x - radius, center_x, 100)
    y_horz = np.full_like(x_horz, center_y)

    return (x_circle, y_circle), (x_vert, y_vert), (x_horz, y_horz)


def plot_7(x_dummy=None, length=3.0, slope=1.5):
    # 1. Diagonal line (/) going upward-right
    x_diag = np.linspace(0, length, 100)
    y_diag = slope * x_diag

    # 2. Horizontal line (-) starting from top endpoint of diagonal line, going left
    x_horz = np.linspace(x_diag[-1], x_diag[-1] - length, 100)
    y_horz = np.full_like(x_horz, y_diag[-1])

    return (x_diag, y_diag), (x_horz, y_horz)

def plot_8(x_dummy=None):
    y = np.linspace(0, 2 * np.pi, 400)
    x1 = np.sin(y)
    x2 = np.cos(np.pi / 2 + y)
    return (x1, y), (x2, y)

def plot_9(x_dummy=None, radius=2.0, center_x=0.0, center_y=0.0):
    # 1. Right half-circle (loop) from -π/2 to π/2
    theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
    x_circle = center_x + radius * np.cos(theta)
    y_circle = center_y + radius * np.sin(theta)

    # 2. Vertical line from bottom to center (|), aligned with rightmost point of circle
    x_vert = np.full(100, center_x )  # Right edge of the circle
    y_vert = np.linspace(center_y + radius, center_y, 100)

    # 3. Horizontal line (-) from top of vertical line to the left (into the circle)
    x_horz = np.linspace(center_x + radius, center_x, 100)
    y_horz = np.full_like(x_horz, center_y)

    return (x_circle, y_circle), (x_vert, y_vert), (x_horz, y_horz)


def plot_a(x_dummy=None, radius=2.0, center_x=0.0, center_y=0.0):
    # 1. Right half-circle (from top to bottom: -π/2 to π/2)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
    x_circle = center_x + radius * np.cos(theta)
    y_circle = center_y + radius * np.sin(theta)

    # 2. Vertical line inside circle (from bottom to center)
    x_inner_vert = np.full(100, center_x )
    y_inner_vert = np.linspace(center_y - radius, center_y, 100)

    # 3. Horizontal line from top of vertical into the circle
    x_horz = np.linspace(center_x + radius, center_x, 100)
    y_horz = np.full_like(x_horz, center_y)

    # 4. Outer vertical line (from bottom to top), aligned at far right of circle
    x_outer_vert = np.full(100, center_x + radius )
    y_outer_vert = np.linspace(center_y - radius, center_y, 100)

    return (x_circle, y_circle), (x_inner_vert, y_inner_vert), (x_horz, y_horz), (x_outer_vert, y_outer_vert)

def plot_b(x, p1=1.0, p2=0.0): 
    return -p1 * x**2 + p2

def plot_c(x, p1=1.0, p2=1.0): 
    return p1 * np.arctan(p2 * x)

def plot_d(x, p1=1.0, p2=1.0): 
    return p1 * np.log(p2 * (x + 3.1))  # avoid log(0)

def plot_e(x, p1=1.0, p2=1.0): 
    return -p1 * x**3 + p2 * x

def plot_f(x, p1=1.0, p2=0.3): 
    return p1 * np.cos(x) + p2 * x

# Mapping hex characters to their plot functions
plot_function_map = {
    '0': plot_0, '1': plot_1, '2': plot_2, '3': plot_3,
    '4': plot_4, '5': plot_5, '6': plot_6, '7': plot_7,
    '8': plot_8, '9': plot_9, 'a': plot_a, 'b': plot_b,
    'c': plot_c, 'd': plot_d, 'e': plot_e, 'f': plot_f
}

# x values for functions expecting x as input
x_vals = np.linspace(-3, 3, 400)

# Main plot rendering function
def draw_plot_group(chars, group_index):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Plot Group {group_index + 1}: Characters {', '.join(chars)}", fontsize=14)

    for i, char in enumerate(chars):
        func = plot_function_map[char]
        ax = axes[i]
        try:
            y_vals = func(x_vals)

            # Handle plot_0 which returns two arrays to plot as parametric curve
            if char == '0':
                x_curve, y_curve = y_vals
                ax.plot(x_curve, y_curve, color='blue', label='Ellipse')
                ax.legend(fontsize=8)

            # plot_1 returns (y_line, (x_vert, y_vert))
            elif char == '1':
                y_line, (x_vert, y_vert) = y_vals
                ax.plot(x_vals, y_line, label='y = mx + c', color='blue')
                ax.plot(x_vert, y_vert, label='x = 1', color='red')
                ax.legend(fontsize=8)

            # plot_2 returns two tuples of arrays (x_parabola, y_vals), (x_base, y_base)
            elif char == '2':
                (x_parabola, y_parabola), (x_base, y_base) = y_vals
                ax.plot(x_parabola, y_parabola, label='Parabola', color='blue')
                ax.plot(x_base, y_base, label=f'y = {y_base[0]:.1f}', color='purple')
                ax.legend(fontsize=8)

            # plot_3 returns rotated points (x_vals, y_vals)
            elif char == '3':
                x_rot, y_rot = y_vals
                ax.plot(x_rot, y_rot, label='Rotated Curve', color='blue')
                ax.legend(fontsize=8)

            # plot_4 returns (y_line, (x_vert, y_vert), (x, y_base))
            elif char == '4':
                y_line, (x_vert, y_vert), (x_base, y_base) = y_vals
                ax.plot(x_vals, y_line, label='y = mx + c', color='blue')
                ax.plot(x_vert, y_vert, label='x = 1', color='red')
                ax.plot(x_base, y_base, label=f'y = {y_base[0]:.1f}', color='purple')
                ax.legend(fontsize=8)

            # plot_5
            elif char == '5':
                (x_top, y_top), (x_vert, y_vert), (x_mid, y_mid), (x_curve, y_curve) = y_vals
                ax.plot(x_top, y_top, label='Top Line', color='blue')
                ax.plot(x_vert, y_vert, label='Left Vertical', color='green')
                ax.plot(x_mid, y_mid, label='Middle Line', color='orange')
                ax.plot(x_curve, y_curve, label='Bottom Curve', color='purple')

            # plot_6
            elif char == '6':
                (x_circle, y_circle), (x_vert, y_vert), (x_horz, y_horz) = y_vals
                ax.plot(x_circle, y_circle, label='Half Circle', color='blue')
                ax.plot(x_vert, y_vert, label='Vertical Line', color='green')
                ax.plot(x_horz, y_horz, label='Connector Line', color='orange')
                ax.legend(fontsize=8)

            # plot_7
            elif char == '7':
                (x_diag, y_diag), (x_horz, y_horz) = y_vals
                ax.plot(x_diag, y_diag, label='Diagonal leg', color='green')
                ax.plot(x_horz, y_horz, label='Top bar', color='blue')
                ax.legend(fontsize=8)

            # plot_8
            elif char == '8':
                (x1, y1), (x2, y2) = y_vals
                ax.plot(x1, y1, label='x = sin(y)', color='blue')
                ax.plot(x2, y2, label='x = cos(π/2 + y)', color='green')
                ax.legend(fontsize=8)

            # plot_9
            elif char == '9':
                (x_circle, y_circle), (x_vert, y_vert), (x_horz, y_horz) = y_vals
                ax.plot(x_circle, y_circle, label='Right half circle', color='blue')
                ax.plot(x_vert, y_vert, label='Vertical line', color='green')
                ax.plot(x_horz, y_horz, label='Horizontal line', color='red')
                ax.legend(fontsize=8)

            # plot_a
            elif char == 'a':
                (x_circle, y_circle), (x_inner_vert, y_inner_vert), (x_horz, y_horz), (x_outer_vert, y_outer_vert) = y_vals
                ax.plot(x_circle, y_circle, label='Right half-circle', color='blue')
                ax.plot(x_inner_vert, y_inner_vert, label='Inner vertical', color='green')
                ax.plot(x_horz, y_horz, label='Top bar', color='red')
                ax.plot(x_outer_vert, y_outer_vert, label='Outer vertical', color='purple')
                ax.legend(fontsize=8)



            # For all other characters: plot y_vals vs x_vals
            else:
                if isinstance(y_vals, tuple):
                    # Defensive check: if function returns tuple (x,y), plot that
                    if len(y_vals) == 2 and all(isinstance(arr, np.ndarray) for arr in y_vals):
                        x_custom, y_custom = y_vals
                        ax.plot(x_custom, y_custom, color='blue')
                    else:
                        # fallback: plot x_vals vs first element of tuple
                        ax.plot(x_vals, y_vals[0], color='blue')
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

# Group characters for plotting
char_groups = [
    ['0', '1', '2', '3'],
    ['4', '5', '6', '7'],
    ['8', '9', 'a', 'b'],
    ['c', 'd', 'e', 'f']
]

# Plot all groups
for idx, group in enumerate(char_groups):
    draw_plot_group(group, idx)
