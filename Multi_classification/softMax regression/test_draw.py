import matplotlib.pyplot as plt
import numpy as np

def draw_lines(coefficients, x_range=(-10, 10), num_points=100):
    """
    Draw multiple lines based on their coefficients in the form ax + by + c = 0.

    Parameters:
    - coefficients: List of tuples, where each tuple contains (a, b, c) for a line.
    - x_range: Tuple (x_min, x_max) specifying the range of x values.
    - num_points: Number of points to generate for x values.

    Example usage:
    coefficients = [(2, -3, 5), (-1, 1, -2), (3, 2, 1)]
    draw_lines(coefficients)
    """
    x = np.linspace(x_range[0], x_range[1], num_points)

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']

    for i, (a, b, c) in enumerate(coefficients):
        y = (-a * x - c) / b
        label = f'{a}x + {b}y + {c} = 0'
        plt.plot(x, y, label=label, color=colors[i])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Multiple Lines Plot')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.show()

# Example usage
coefficients = [(2, -3, 5), (-1, 1, -2), (3, 2, 1)]
draw_lines(coefficients)
