import numpy as np
import matplotlib.pyplot as plt
def check_dom(a):
    n = len(a)
    for i in range(n):
        row_sum = np.sum(np.abs(a[i])) - np.abs(a[i][i])
        if np.abs(a[i][i]) < row_sum:
            return False
    return True
def gauss_seidel(a, b, tol=0.009, max_iter=100, figsize=(15, 10)):
    if not check_dom(a):
        print("Error: Matrix is not diagonally dominant")
    n = len(b)
    x = np.zeros(n)
    history = [x.copy()]
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            if a[i, i] == 0:
                print(f"Zero diagonal element at index {i}.")
            s1 = np.dot(a[i, :i], x_new[:i])
            s2 = np.dot(a[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / a[i, i]
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {k+1} iterations.")
            break
        x = x_new
    else:
        print(f"Reached max iterations ({max_iter}).")
    solution = np.round(x, 3)
    x_hist = np.array(history)
    iters = np.arange(x_hist.shape[0])
    plt.figure(figsize=figsize)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    markers = ['o', 's', '^']
    for idx in range(n):
        ax.plot(iters, x_hist[:, idx], marker=markers[idx], markersize=8, 
                linewidth=3, alpha=0.85, label=f"x[{idx}]" )
        final_val = x_hist[-1, idx]
        ax.scatter(iters[-1], final_val, marker='X', s=120)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Value of x", fontsize=14)
    plt.title("Gauss–Seidel Method", fontsize=16, pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, frameon=True, loc='best')
    plt.tight_layout()
    plt.show()
    return solution
if __name__ == '__main__':
    a = np.array([
            [6, 1, 0],
            [1, 3, 1],
            [0, 1, 5]
        ])
    b = np.array([7, 5, 6])
    solution = gauss_seidel(a, b, tol=0.011, max_iter=100)
    print("Solution :", solution)



#=======================================================================================

# import numpy as np
# import matplotlib.pyplot as plt
# def check_dom(a):
#     n = len(a)
#     for i in range(n):
#         row_sum = np.sum(np.abs(a[i])) - np.abs(a[i][i])
#         if np.abs(a[i][i]) < row_sum:
#             return False
#     return True
# def gauss_seidel(a, b, tol=0.005, max_iter=100, figsize=(15, 10)):
#     if not check_dom(a):
#         print("Error: Matrix is not diagonally dominant")

#     n = len(b)
#     x = np.zeros(n)
#     history = [x.copy()]

#     for k in range(max_iter):
#         x_new = x.copy()
#         for i in range(n):
#             if a[i, i] == 0:
#                 print(f"Zero diagonal element at index {i}.")
#             s1 = np.dot(a[i, :i], x_new[:i])
#             s2 = np.dot(a[i, i+1:], x[i+1:])
#             x_new[i] = (b[i] - s1 - s2) / a[i, i]

#         history.append(x_new.copy())
#         if np.linalg.norm(x_new - x) < tol:
#             print(f"Converged in {k+1} iterations.")
#             break
#         x = x_new
#     else:
#         print(f"Reached max iterations ({max_iter}).")

#     x_hist = np.array(history)
#     iters = np.arange(x_hist.shape[0])

#     # Create plot
#     plt.figure(figsize=figsize)
#     ax = plt.gca()
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     for spine in ['left', 'bottom']:
#         ax.spines[spine].set_linewidth(1.2)

#     markers = ['o', 's', '^', 'D', 'v']
#     for idx in range(x_hist.shape[1]):
#         line, = ax.plot(
#             iters,
#             x_hist[:, idx],
#             marker=markers[idx % len(markers)],
#             markersize=8,
#             linewidth=3,
#             alpha=0.85,
#             label=f"x[{idx}]"
#         )
#         final_val = x_hist[-1, idx]
#         ax.scatter(iters[-1], final_val, marker='X', s=120)
#     plt.xlabel("Iteration", fontsize=14)
#     plt.ylabel("Value of x", fontsize=14)
#     plt.title("Gauss–Seidel Method", fontsize=16, pad=15)
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(fontsize=12, frameon=True,loc='best')
#     plt.text(0.98, 0.02,'Suhan Khadka  [NCE080BCT043]',ha='right', 
#         va='bottom', transform=ax.transAxes,fontsize=16, 
#         bbox=dict(facecolor='white', edgecolor='black', pad=3)
#     )
#     plt.tight_layout()
#     plt.show()
#     return x

# if __name__ == '__main__':
#     a = np.array([
#         [4, 1, 2],
#         [3, 5, 1],
#         [1, 1, 3]
#     ])
#     b = np.array([4, 7, 3])
#     solution = gauss_seidel(a, b, tol=0.005, max_iter=100)
#     print("Solution :", solution)

#=======================================================================================

# #gauss seidal
# import numpy as np
# import matplotlib.pyplot as plt

# a = np.array([
#     [4, 1, 2],
#     [3, 5, 1],
#     [1, 1, 3]
# ])
# b = np.array([4, 7, 3])

# def check_dom(a):
#     n = len(a)
#     for i in range(n):
#         row_sum = np.sum(np.abs(a[i])) - np.abs(a[i][i])
#         if np.abs(a[i][i]) < row_sum:
#             return False
#     return True


# def gauss_seidal(a,b,tol=0.005,max_iter = 100):
#   if check_dom(a):
#     n = len(b)
#     x = np.zeros(n)
#     x_prev = [x.copy()]
#     x_new = x.copy()  # Ensure x_new is always defined
#     k = -1            # Initialize k in case the loop does not run
#     for k in range(max_iter):
#       x_new = x.copy()
#       for i in range(n):
#         s1 = 0
#         s2 = 0
#         if a[i][i] == 0:
#           print(f"Zero diagonal element at index {i}, which will cause division by zero.")
#         for j in range(i):
#           s1 = s1 + (a[i][j] * x_new[j])
#         for j in range(i+1,n):
#           s2 = s2 + (a[i][j] * x[j])
#         x_new[i] = (b[i] - s1 - s2) / a[i][i]

#       x_prev.append(x_new.copy())
#       if np.linalg.norm(x_new - x) < tol:
#         break
#       x = x_new

#     # If the loop never runs, k will be -1, so set iteration accordingly
#     iteration = k + 1 if k >= 0 else 0
#     solution, x_hist = x_new, np.array(x_prev)

#     print("Solution : ", solution)
#     print("Iteration : ", iteration)
#     iters = np.arange(x_hist.shape[0])
#     plt.figure(figsize=(8, 5))

#     for idx in range(x_hist.shape[1]):
#         plt.plot(iters, x_hist[:, idx], marker='o', label=f"x[{idx}]")

#     plt.xlabel("Iteration")
#     plt.ylabel("Value of x")
#     plt.title("Gauss–Seidel Convergence of Each Component")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# gauss_seidal(a,b)