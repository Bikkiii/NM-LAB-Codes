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

#--------------------------------------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# f = lambda x, y: x + y
# def exact_solution(x):
#     return 2 * np.exp(x) - x - 1 
# def euler_method(f, x0, y0, h, n):
#     x, y = x0, y0
#     xs, ys = [x], [y]
#     for _ in range(n):
#         y += h * f(x, y)
#         x += h
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)
# def rk2_method(f, x0, y0, h, n):
#     x, y = x0, y0
#     xs, ys = [x], [y]
#     for _ in range(n):
#         k1 = f(x, y)
#         k2 = f(x + h, y + h * k1)
#         y += (h / 2) * (k1 + k2)
#         x += h
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)
# def rk4_method(f, x0, y0, h, n):
#     x, y = x0, y0
#     xs, ys = [x], [y]
#     for _ in range(n):
#         k1 = f(x, y)
#         k2 = f(x + h/2, y + h*k1/2)
#         k3 = f(x + h/2, y + h*k2/2)
#         k4 = f(x + h, y + h*k3)
#         y += (h/6) * (k1 + 2*k2 + 2*k3 + k4)
#         x += h
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)
# x0, y0, h, n = 0, 1, 0.1, 20
# xe, ye = euler_method(f, x0, y0, h, n)
# x2, y2 = rk2_method(f, x0, y0, h, n)
# x4, y4 = rk4_method(f, x0, y0, h, n)
# x_ex = np.linspace(x0, x0 + n*h, 300)
# y_ex = exact_solution(x_ex)
# plt.figure(figsize=(10, 6))
# plt.plot(x_ex, y_ex, color='purple', linewidth=3, label='Exact Solution')
# plt.plot(xe, ye, marker='s', linestyle='--', linewidth=1.5, label="Euler's Method")
# plt.plot(x2, y2, marker='D', linestyle='-.', linewidth=1.5, label="RK2 Method")
# plt.plot(x4, y4, marker='o', linestyle=':', linewidth=1.5, label="RK4 Method")
# plt.title("Numerical Solutions for $\\frac{dy}{dx}=x + y$, $y(0)=1$  [Sadeep Khanal]", 
#     fontsize=16, weight='bold', pad=12)
# plt.xlabel("x", fontsize=14)
# plt.ylabel("y", fontsize=14)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=12, frameon=True, edgecolor='gray')
# plt.tight_layout()
# plt.show()