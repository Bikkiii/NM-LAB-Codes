 #LAGRANGE METHOD
import numpy as np
import matplotlib.pyplot as plt
x_p = np.array([1, 2.6, 2, 3, 5])
y_p = np.array([2, 3.1, 1.8, 4.2, 6.8])
def lagrange_poly(x, x_data, y_data):
    total = 0
    n = len(x_data)
    for j in range(n):
        term = y_data[j]
        for m in range(n):
            if m != j:
                term *= (x - x_data[m]) / (x_data[j] - x_data[m])
        total += term
    return total
x_interp = 2.5
interp_val = lagrange_poly(x_interp, x_p, y_p)
print(f"Interpolated Point at x={x_interp} is:", interp_val)
x_vals = np.linspace(-0.5, 5, 400)
y_vals = [lagrange_poly(x, x_p, y_p) for x in x_vals]
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='Lagrange Polynomial', color='#1f77b4', linewidth=2.5)
plt.plot(x_p, y_p, 'ro', markersize=8, label='Data Points')
plt.scatter(2.5, interp_val, color='green', s=100, zorder=5,
            label=f'Interpolated Point\nx=2.5, y≈{interp_val:.2f}')
plt.axvline(2.5, color='green', linestyle='--', linewidth=1)
plt.axhline(interp_val, color='green', linestyle='--', linewidth=1)
plt.title('Lagrange Interpolation Polynomial', fontsize=16, weight='bold')
plt.xlabel('x', fontsize=13)
plt.ylabel('y', fontsize=13)
plt.legend(loc='best', fontsize=11, frameon=True,
           fancybox=True, shadow=True, borderpad=1)
plt.grid(True, linestyle=':', alpha=0.8)
plt.tight_layout()
plt.show()


# #------------------------------------------------------------------------------------

# # #NEWTON DIVIDED DIFFERENCE METHOD
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # def newton_forward_interpolation(x, y, value):
# #     n = len(x)
# #     diff_table = np.zeros((n, n))
# #     diff_table[:, 0] = y
# #     for j in range(1, n):
# #         for i in range(n - j):
# #             diff_table[i, j] = diff_table[i+1, j-1] - diff_table[i, j-1]
# #     cols = ['Δ^{}'.format(i) for i in range(n)]
# #     df = pd.DataFrame(diff_table, columns=cols, index=[f"x={xi}" for xi in x])
# #     print("\nForward Difference Table:")
# #     print(df.round(4))
# #     h = x[1] - x[0]
# #     p = (value - x[0]) / h
# #     yp = y[0]
# #     fact = 1
# #     p_term = 1
# #     for i in range(1, n):
# #         p_term *= (p - (i - 1))
# #         fact *= i
# #         yp += (p_term * diff_table[0, i]) / fact
# #     return diff_table, yp
# # def plot_interpolation(x, y, diff_table, value, yp):
# #     h = x[1] - x[0]
# #     x_plot = np.linspace(x[0], x[-1], 500)
# #     y_plot = []
# #     n = len(x)
# #     for xv in x_plot:
# #         p_temp = (xv - x[0]) / h
# #         yp_temp = y[0]
# #         fact_temp = 1
# #         p_term_temp = 1
# #         for i in range(1, n):
# #             p_term_temp *= (p_temp - (i - 1))
# #             fact_temp *= i
# #             yp_temp += (p_term_temp * diff_table[0, i]) / fact_temp
# #         y_plot.append(yp_temp)
# #     plt.figure(figsize=(10, 6))
# #     plt.plot(x_plot, y_plot, label='Interpolation Polynomial', linewidth=2)
# #     plt.scatter(x, y, color='red', label='Data Points', zorder=5)
# #     plt.scatter([value], [yp], color='green',
# #                 label=f'Interpolated y({value}) = {yp:.4f}', zorder=6)
# #     plt.axvline(value, color='green', linestyle='--', linewidth=1.5)
# #     plt.axhline(yp,    color='green', linestyle='--', linewidth=1.5)
# #     plt.title('Newton Forward Interpolation', fontsize=14, pad=12)
# #     plt.xlabel('x', fontsize=12)
# #     plt.ylabel('y', fontsize=12)
# #     plt.grid(True, linestyle=':', alpha=0.7)
# #     plt.text(0.98, 0.02, 'Suhan Khadka [NCE080BCT043]',
# #       fontsize=12, ha='right', va='bottom', transform=plt.gca().transAxes,
# #       bbox=dict(facecolor='lightyellow', edgecolor='black', boxstyle='round,pad=0.4'))
# #     plt.legend(loc='best', fontsize=11, frameon=True,
# #            fancybox=True, shadow=True, borderpad=1)
# #     plt.tight_layout()
# #     plt.show()
# # if __name__ == "__main__":
# #     x = np.array([2, 4, 6, 8, 10, 12, 14], dtype=float)
# #     y = np.array([7.1, 4.2, 3.9, 4.5, 6.4, 7.5, 5.7], dtype=float)
# #     value_to_interpolate = 3.0
# #     diff_table, yp = newton_forward_interpolation(x, y, value_to_interpolate)
# #     print(f"\nInterpolated result: y({value_to_interpolate}) = {yp:.4f}\n")
# #     plot_interpolation(x, y, diff_table, value_to_interpolate, yp)


#---------------------------------------------------------------------------------------

#NEWTON DIVIDED DIFFERENCE METHOD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def newton_forward(x, y, xi):
    n = len(x)
    D = np.zeros((n, n))
    D[:, 0] = y
    for j in range(1, n):
        D[:n-j, j] = D[1:n-j+1, j-1] - D[:n-j, j-1]
    cols = ['Δ^{}'.format(i) for i in range(n)]
    df = pd.DataFrame(D, columns=cols, index=[f"x={xi}" for xi in x])
    print("\nForward Difference Table:\n")
    print(df.round(4))
    h = x[1] - x[0]
    p = (xi - x[0]) / h
    yp = y[0]
    fact = 1
    p_term = 1
    for i in range(1, n):
        p_term *= p - (i-1)
        fact *= i
        yp += (p_term * D[0, i]) / fact
    return D, yp
def plot_newton(x, y, D, xi, yi):
    h = x[1] - x[0]
    n = len(x)
    xg = np.linspace(x[0], x[-1], 500)
    yg = []
    for xv in xg:
        p = (xv - x[0]) / h
        val = y[0]
        fact = 1
        p_term = 1
        for i in range(1, n):
            p_term *= p - (i-1)
            fact *= i
            val += (p_term * D[0, i]) / fact
        yg.append(val)
    plt.figure(figsize=(10,6))
    plt.plot(xg, yg, lw=2, label='Interpolation Polynomial')
    plt.scatter(x, y, c='red', zorder=5, label='Data Points')
    plt.scatter(xi, yi, c='green', zorder=6, label=f'y({xi})={yi:.4f}')
    plt.axvline(xi, ls='--', c='green')
    plt.axhline(yi, ls='--', c='green')
    plt.title('Newton Forward Interpolation')
    plt.xlabel('x'); plt.ylabel('y')
    plt.grid(linestyle=':', alpha=0.7)
    plt.legend(loc='best', fontsize=11, frameon=True,fancybox=True, shadow=True, borderpad=1) 
    plt.tight_layout(); plt.show()
if __name__ == "__main__":
    x = np.array([2,4,6,8,10,12,14], float)
    y = np.array([7.1,4.2,3.9,4.5,6.4,7.5,5.7], float)
    xi = 3.0
    D, yi = newton_forward(x, y, xi)
    print(f"\nInterpolated result: y({xi}) = {yi:.4f}\n")
    plot_newton(x, y, D, xi, yi)
