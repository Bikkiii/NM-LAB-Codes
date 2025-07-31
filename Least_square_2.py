#Exponential Fit of Least Squares
import numpy as np
import matplotlib.pyplot as plt
x = np.array([0.3, 0.9, 1.6, 2.2, 2.9, 3.5, 4.1], dtype=float)
y = np.array([1.05, 1.65, 2.75, 4.10, 6.20, 9.30, 13.90], dtype=float)
ln_y = np.log(y)
n = len(x)
sum_x = np.sum(x)
sum_ln = np.sum(ln_y)
sum_xln = np.sum(x * ln_y)
sum_x2 = np.sum(x * x)
B = (n * sum_xln - sum_x * sum_ln) / (n * sum_x2 - sum_x ** 2)
lnA = (sum_ln - B * sum_x) / n
A = np.exp(lnA)
y_pred = A * np.exp(B * x)
x_vals = np.linspace(x.min(), x.max(), 200)
y_vals = A * np.exp(B * x_vals)
plt.figure(figsize=(8, 5), facecolor='white')
plt.scatter(x, y, s=80, edgecolor='k', label='Data Points', zorder=5)
plt.plot(x_vals, y_vals, linewidth=2.5, label=rf'$y = {A:.3f}e^{{{B:.3f}x}}$')
plt.title("Exponential Fit: $y = Ae^{Bx}$", fontsize=16, weight='bold', pad=12)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


#------------------------------------------------------------------------------------------


#Poly_fit of Least Squares Method
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import det
def poly_fit(x, y, degree):
    m = degree + 1
    A = np.zeros((m, m))
    B = np.zeros(m)
    for i in range(m):
        B[i] = np.sum(x**i * y)
        for j in range(m):
            A[i, j] = np.sum(x**(i + j))
    D = det(A)
    if np.isclose(D, 0):
        print("Singular matrix – cannot solve via Cramer's rule.")
    coeffs = np.zeros(m)
    for i in range(m):
        Ai = A.copy()
        Ai[:, i] = B
        coeffs[i] = det(Ai) / D
    return coeffs
def plot_poly_fit(x, y, degree):
    coeffs = poly_fit(x, y, degree)
    x_vals = np.linspace(x.min(), x.max(), 300)
    y_vals = sum(coef * x_vals**i for i, coef in enumerate(coeffs))
    terms = [f"{coeffs[0]:.3f}"]
    for i in range(1, degree + 1):
        terms.append(f"{coeffs[i]:+.3f}x^{i}")
    formula = " ".join(terms)
    label = rf"$y = {formula}$"
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.scatter(x, y, s=80, edgecolor='black', label='Data Points', zorder=5)
    ax.plot(x_vals, y_vals, '.-', linewidth=1.5, label=label)
    ax.set_title(f"{degree}‑Degree Polynomial Curve Fit", fontsize=16, weight='bold', pad=12)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    ax.legend(fontsize=12, frameon=True, edgecolor='gray')
    plt.tight_layout()
    plt.show()
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
y = np.array([2.00, 3.35, 4.50, 5.45, 6.00, 5.75, 5.20, 4.15, 2.80, 1.15])
degree = 4
plot_poly_fit(x, y, degree)
