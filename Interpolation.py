import numpy as np
import matplotlib.pyplot as plt

def Lagrange(x, y, x0):
  n = len(x)
  y0 = 0.0

  for i in range(n):
    prod = y[i]
    for j in range(n):
      if i != j:
        prod *= (x0 - x[j]) / (x[i] - x[j])
    y0 += prod
  return y0

xVals = np.array([2, 4, 6, 8, 10, 12, 14], dtype=float)
yVals = np.array([7.1, 4.8, 3.9, 4.5, 6.4, 7.5, 5.7])

# xPoint = 9.0
xPoint = np.array([5.0, 7.0, 11.0])
yPoint = Lagrange(xVals, yVals, xPoint)
print(yPoint)

x0 = min(xVals)
xn = max(xVals)
X = np.linspace(x0,xn,200)
Y = Lagrange(xVals, yVals, X)
plt.figure(figsize=(8, 4))
plt.grid()
plt.plot(X,Y, color="blue", label="Line")
plt.scatter(xVals, yVals, color="black", label="Given")
plt.scatter(xPoint, yPoint, color="red", label="Interpolated")
plt.legend()
plt.show()