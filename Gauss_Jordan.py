# GAUSS–JORDAN METHOD 
import numpy as np
def gauss_jordan(a, b, tol=1e-12):
    A = np.array(a, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n = A.shape[0]
    aug = np.hstack((A, b))
    for i in range(n):
        max_row = np.argmax(np.abs(aug[i:, i])) + i
        if abs(aug[max_row, i]) < tol:
            print("Matrix is singular or nearly singular.")
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]
        aug[i] /= aug[i, i]
        for j in range(n):
            if j != i:
                aug[j] -= aug[j, i] * aug[i]
    header = [f"a{j+1}" for j in range(n)] + ["b"]
    print("\nFinal Augmented Matrix [A | b]:")
    print("  " + "  ".join(f"{h:>8}" for h in header))
    print("   " + "-" * (11*(n+1) - 2))
    for row in aug:
        print("   " + "   ".join(f"{val:8.5f}" for val in row))
    return aug[:, -1]
if __name__ == "__main__":
    A = [
        [ 2,  1, -1],
        [-3, -1,  2],
        [-2,  1,  2]
    ]
    B = [8, -11, -3]
    solution = gauss_jordan(A, B)
    print("\nSolution:")
    for i, val in enumerate(solution, start=1):
        print(f"  x{i:<2} = {val:8.5f}")


#-----------------------------------------------------------

# Guass Elimination Method
import numpy as np
def gaussian_elimination(a, b, tol=1e-12):
    A = np.array(a, float)
    b = np.array(b, float).reshape(-1, 1)
    n = A.shape[0]
    aug = np.hstack((A, b))
    for i in range(n):
        max_row = np.argmax(np.abs(aug[i:, i])) + i
        if abs(aug[max_row, i]) < tol:
            raise ValueError("Matrix is singular or nearly singular.")
        if max_row != i:
            aug[[i, max_row]] = aug[[max_row, i]]
        for j in range(i+1, n):
            factor = aug[j, i] / aug[i, i]
            aug[j, i:] -= factor * aug[i, i:]
    print("\n" + "="*60)
    print("     Upper‑Triangular Augmented Matrix [ U | c ]")
    print("="*60)
    header = [f"u{j+1}" for j in range(n)] + ["   c"]
    print(" " + "  ".join(f"{h:>8}" for h in header))
    print("  " + "-"*(11*(n+1)-4))
    for row in aug:
        print("  " + "  ".join(f"{val:8.5f}" for val in row))
    print("="*60 + "\n")
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        rhs = aug[i, -1] - np.dot(aug[i, i+1:n], x[i+1:n])
        x[i] = rhs / aug[i, i]
    return x
if __name__ == "__main__":
    A = [
        [ 4.00,  -2.00,   1.00,   3.00],
        [ 3.00,   6.00,  -4.00,   2.00],
        [ 2.00,   1.00,   8.00,  -5.00],
        [ 1.00,   3.00,  -2.00,   6.00]
    ]
    B = [5.0, 9.0, 4.0, 3.0]
    solution = gaussian_elimination(A, B)
    print("   Solution Vector")
    print("="*60)
    for i, val in enumerate(solution, start=1):
        print(f"   x{i:<2} = {val:12.5f}")
    print("="*60)
