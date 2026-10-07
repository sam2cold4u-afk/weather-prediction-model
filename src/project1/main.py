from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def compute_error(b, m, points):
    x = points[:, 0]
    y = points[:, 1]
    # MSE: mean of squared differences
    return np.mean((y - (m * x + b)) ** 2)

def step_gradient(b_current, m_current, points, learning_rate):
    x = points[:, 0]
    y = points[:, 1]
    N = float(len(points))
    
    # Vectorized gradient calculation
    y_pred = m_current * x + b_current
    error = y - y_pred
    
    b_gradient = -(2 / N) * np.sum(error)
    m_gradient = -(2 / N) * np.sum(x * error)
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for _ in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return b, m

def run():
    # Resolves data.csv in the same folder as this script
    data_path = Path(__file__).resolve().parent / "data.csv"
    points = np.genfromtxt(data_path, delimiter=",")
    
    learning_rate = 0.0001
    initial_b = 0.0
    initial_m = 0.0
    num_iterations = 1000
    
    initial_error = compute_error(initial_b, initial_m, points)
    print(f"Starting at b = {initial_b:.4f}, m = {initial_m:.4f}, MSE = {initial_error:.4f}")
    
    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    
    final_error = compute_error(b, m, points)
    print(f"Finished {num_iterations} iterations: b = {b:.4f}, m = {m:.4f}, MSE = {final_error:.4f}")

    # Plot results
    plt.scatter(points[:, 0], points[:, 1], color="blue", label="Actual Data")
    x_vals = points[:, 0]
    plt.plot(x_vals, m * x_vals + b, color="red", label=f"Model: y = {m:.2f}x + {b:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression via Gradient Descent")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run()