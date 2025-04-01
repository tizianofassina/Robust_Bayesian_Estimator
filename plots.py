#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Define the functions using indicator functions
# -----------------------------------------------------------------------------

def f1(x, q=0.5, x0=0):
    """
    Function: q * 1_{(x0, ∞)}(x)
    Returns q for x > x0, and 0 for x <= x0.
    """
    return np.where(x > x0, q, 0)

def f2(x, q=0.5, x0=0):
    """
    Function: 1_{[x0, ∞)}(x) + q * 1_{(-∞, x0)}(x)
    Returns 1 for x >= x0 and q for x < x0.
    """
    return np.where(x >= x0, 1, q)

def f3(x):
    """
    Application 1 (Hydrology) - Infimum function:
      0.05 * 1_{[1250, 2000)}(x) +
      0.7317 * 1_{[2000, 2100)}(x) +
      0.75 * 1_{[2100, ∞)}(x)
    Returns 0 for x < 1250.
    """
    return np.piecewise(
        x,
        [x < 1250, (x >= 1250) & (x < 2000), (x >= 2000) & (x < 2100), x >= 2100],
        [0, 0.05, 0.7317, 0.75]
    )

def f4(x):
    """
    Application 1 (Hydrology) - Supremum function:
      0.05 * 1_{(-∞, 1250)}(x) +
      0.06834 * 1_{[1250, 2000)}(x) +
      0.75 * 1_{[2000, 2100)}(x) +
      1.0 * 1_{[2100, ∞)}(x)
    """
    return np.piecewise(
        x,
        [x < 1250, (x >= 1250) & (x < 2000), (x >= 2000) & (x < 2100), x >= 2100],
        [0.05, 0.06834, 0.75, 1.0]
    )

def f5(x):
    """
    Application 2 (Meteorology) - Infimum function:
      0.25 * 1_{[75, 100)}(x) +
      0.5 * 1_{[100, 150)}(x) +
      0.75 * 1_{[150, ∞)}(x)
    Returns 0 for x < 75.
    """
    return np.piecewise(
        x,
        [x < 75, (x >= 75) & (x < 100), (x >= 100) & (x < 150), x >= 150],
        [0, 0.25, 0.5, 0.75]
    )

def f6(x):
    """
    Application 2 (Meteorology) - Supremum function:
      0.25 * 1_{(-∞, 75)}(x) +
      0.5 * 1_{[75, 100)}(x) +
      0.75 * 1_{[100, 150)}(x) +
      1.0 * 1_{[150, ∞)}(x)
    """
    return np.piecewise(
        x,
        [x < 75, (x >= 75) & (x < 100), (x >= 100) & (x < 150), x >= 150],
        [0.25, 0.5, 0.75, 1.0]
    )

# -----------------------------------------------------------------------------
# Utility function to plot a given function over a specified range
# -----------------------------------------------------------------------------

def plot_function(func, x_range, title, xlabel="", ylabel="", show=True, name="",
                  grid=True, custom_xticks=None, custom_yticks=None,
                  custom_xtick_labels=None, custom_ytick_labels=None,
                  xtick_rotation=0, ytick_rotation=0, label="", legend=False):
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = func(x)
    plt.step(x, y, where='post', linewidth=2, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.ylim(min(y)-0.1, max(y)+0.1)
    
    # Set custom ticks if provided, along with rotation
    if custom_xticks is not None:
        if custom_xtick_labels is not None:
            plt.xticks(custom_xticks, custom_xtick_labels, rotation=xtick_rotation)
        else:
            plt.xticks(custom_xticks, rotation=xtick_rotation)
    if custom_yticks is not None:
        if custom_ytick_labels is not None:
            plt.yticks(custom_yticks, custom_ytick_labels, rotation=ytick_rotation)
        else:
            plt.yticks(custom_yticks, rotation=ytick_rotation)
            
    if legend:
        plt.legend()
    if name != "":
        plt.tight_layout()
        plt.savefig(name, dpi=300)
    if show:
        plt.tight_layout()
        plt.show()

# -----------------------------------------------------------------------------
# Main function: plot all indicator-based functions
# -----------------------------------------------------------------------------

def main():
    sns.set_theme(style="dark", rc={
    "axes.edgecolor": "black",    # white borders
    "axes.spines.top": True,        # show top spine
    "axes.spines.right": True,      # show right spine
})
    
    # Plot f1: q * 1_{(x0, ∞)}(x) with q=0.5 and x0=0
    # For the first plot, only show x=0 (labeled as x_0) and y-ticks at 0, 0.5, 1.
    plt.figure(figsize=(4, 2))
    plot_function(
        lambda x: f1(x, q=0.5, x0=0),
        (-10, 10),
        r"$q 1_{(x_0,\infty)}(x)$",
        custom_xticks=[0],
        custom_xtick_labels=[r"$x_0$"],
        custom_yticks=[0, 0.5, 1],
        custom_ytick_labels=[0, r"$q$", 1],
        name="lemma_1_function"
    )
    
    # Plot f2: 1_{[x0, ∞)}(x) + 0.5 * 1_{(-∞, x0)}(x) with x0=0
    # For the second plot, use the same custom tick settings.
    plt.figure(figsize=(4, 2))
    plot_function(
        lambda x: f2(x, q=0.5, x0=0),
        (-10, 10),
        r"$1_{[x_0,\infty)}(x) + q 1_{(-\infty,x_0)}(x)$",
        custom_xticks=[0],
        custom_xtick_labels=[r"$x_0$"],
        custom_yticks=[0, 0.5, 1],
        custom_ytick_labels=[0, r"$q$", 1],
        name="lemma_2_function"
    )
    
    # Application 1: Hydrology functions
    # For these plots, show only the points where the function changes.
    # For f3, the x–ticks are at the breakpoints and y–ticks are the unique function values.
    plt.figure(figsize=(10, 5))
    plot_function(
        f3,
        (1000, 2500),
        "Application 1 (Hydrology)",
        show=False,
        custom_xticks=[1250, 2000, 2100],
        custom_yticks=[0, 0.05, 0.7317, 0.75],
        xtick_rotation=45,
        label="inf"
    )
    # For f4, set ticks at the change points.
    plot_function(
        f4,
        (1000, 2500),
        "Application 1 (Hydrology)",
        custom_xticks=[1250, 2000, 2100],
        custom_yticks=[0, 0.04, 0.08, 0.7, 0.75, 1.0],
        custom_ytick_labels=[0, 0.05, 0.068, 0.73, 0.75, 1.0],
        xtick_rotation=45,
        grid=False,
        label="sup",
        legend=True,
        name="application1"
    )
    
    # Application 2: Meteorology functions remain unchanged
    plt.figure(figsize=(8, 4))
    plot_function(
        f5,
        (50, 200),
        "Application 2 (Meteorology)",
        show=False,
        custom_xticks=[75, 100, 150],
        custom_yticks=[0, 0.25, 0.5, 0.75, 1.0],
        label="inf"
    )
    plot_function(
        f6,
        (50, 200),
        "Application 2 (Meteorology)",
        custom_xticks=[75, 100, 150],
        custom_yticks=[0, 0.25, 0.5, 0.75, 1.0],
        label="sup",
        legend=True,
        name="application2"
    )

# -----------------------------------------------------------------------------
# Run the script
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
