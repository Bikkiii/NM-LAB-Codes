import numpy as np
import matplotlib.pyplot as plt

def Fitting(x,y):
  X = np.array(x)
  Y = np.array(y)
  sumx = np.sum(X)
  sumy = np.sum(Y)
  sumx2 = np.sum(X*X)
  sumxy = np.sum(X*Y)

  b = (sumxy + sumx*sumy)/(sumx*sumx + sumx2)
  a = b * sumx - sumy

  return a,b


# for linear eqn y = a + bx
xVals = [2, 4, 6, 8, 10, 12]
yVals = [6, 7, 9, 10, 14, 15]
a,b = Fitting(xVals,yVals)
print("a = ", a , ", b = ",b)

x0 = min(xVals)
xn = max(yVals)
X = np.linspace(x0,xn)
Y = a + b * X
plt.figure(figsize=(8,8))
plt.grid()
plt.plot(X,Y,color="blue", label="best fit")
plt.scatter(xVals, yVals, color="black", label = "Given Points")
plt.legend()
plt.show()


# for exponent eqn : y = a e**(bx)
def f(a,b,x):
  return a * np.exp(b*x)

xVals = [2, 4, 6, 8, 10, 12]
yVals = [5, 8, 15, 32, 50, 120]
yLog = np.log(yVals)

A, b = Fitting(xVals, yLog)
a = np.exp(A)

print("y = ", a," * exp( ", b, " * x )")

x0 = min(xVals)
xn = max(xVals)
X = np.linspace(x0,xn)
Y = f(a,b,X)
plt.figure(figsize=(8,8))
plt.grid()
plt.plot(X,Y,color="blue", label="best fit")
plt.scatter(xVals, yVals, color="black", label = "Given Points")
plt.legend()
plt.show()