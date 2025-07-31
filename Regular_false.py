def f(x):
    return x**3 + 3*x**2 - 5

def regula_falsi_modified(a, b, tolerance=1e-5, max_iterations=100):
    if f(a) * f(b) >= 0:
        print("Invalid initial interval. f(a) and f(b) must have opposite signs.")
        return

    # Table header
    print(f"{'Iter':<8}{'a':<12}{'b':<12}{'c':<12}{'f(c)':<14}")
    print("-" * 58)
import numpy as np
import matplotlib.pyplot as plt
f = lambda x, y: x - y
def exact_solution(x):
    return x - 1 + 3 * np.exp(-x)
def euler_method(f, x0, y0, h, n):
    x, y = x0, y0
    xs, ys = [x], [y]
    for _ in range(n):
        y += h * f(x, y)
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
def rk2_method(f, x0, y0, h, n):
    x, y = x0, y0
    xs, ys = [x], [y]
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h, y + h * k1)
        y += (h / 2) * (k1 + k2)
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
def rk4_method(f, x0, y0, h, n):
    x, y = x0, y0
    xs, ys = [x], [y]
    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h/2, y + h*k1/2)
        k3 = f(x + h/2, y + h*k2/2)
        k4 = f(x + h, y + h*k3)
        y += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
x0, y0, h, n = 0, 2, 0.1, 20
xe, ye = euler_method(f, x0, y0, h, n)
x2, y2 = rk2_method(f, x0, y0, h, n)
x4, y4 = rk4_method(f, x0, y0, h, n)
x_ex = np.linspace(x0, x0 + n*h, 300)
y_ex = exact_solution(x_ex)
plt.figure(figsize=(10, 6))
plt.plot(x_ex, y_ex, color='purple', linewidth=3, label='Exact Solution')
plt.plot(xe, ye, marker='s', linestyle='--', linewidth=1.5, label="Euler's Method")
plt.plot(x2, y2, marker='D', linestyle='-.', linewidth=1.5, label="RK2 Method")
plt.plot(x4, y4, marker='o', linestyle=':', linewidth=1.5, label="RK4 Method")
plt.title("Numerical Solutions for $\\frac{dy}{dx}=x - y$, $y(0)=2$", 
    fontsize=16, weight='bold', pad=12)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12, frameon=True, edgecolor='gray')
plt.tight_layout()
plt.show()

max_iterations=100
tolerance=0.00005
for i in range(1, max_iterations + 1):
        fa = f(a)
        fb = f(b)

        # Regula Falsi formula (alternate form)
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)

        # Print current row
        print(f"{i:<8}{a:<12.5f}{b:<12.5f}{c:<12.5f}{fc:<14.5f}")

        # Check for convergence
        if abs(fc) < tolerance:
            print("\nRoot found with desired accuracy.")
            break

        # Update interval
        if fa * fc < 0:
            b = c
        else:
            a = c

        # Check if c has converged to 5 decimal digits
        if round(prev_c, 5) == round(c, 5):
            print("\nRoot converged to 5 decimal places.")
            break

        prev_c = c

# Run the method on interval [1, 2]
regula_falsi_modified(1, 2)
