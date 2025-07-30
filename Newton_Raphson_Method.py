import numpy as np
import matplotlib.pyplot as plt

def NR(F, G, x0):
  tol = 0.00005

  if abs(F(x0)) < tol:
    return round(x0,5)

  for i in range(100):
    if abs(G(x0)) < tol:
      print("Error! Division by ~zero.")
      return None

    x1 = x0 - (F(x0)/G(x0))

    if abs(x1-x0) < tol:
      return round(x0,5)

    x0 = x1
  return None

def F(x):
  return x * np.sin(x) + np.cos(x)

def G(x):
  return np.cos(x) * x + np.sin(x) - np.sin(x)

root = NR(F,G,1)
print("The root = ", root)

plt.figure(figsize=(9,2.5))
X = np.linspace(1,root,210)
Y = F(X)
plt.grid()
plt.plot(X,Y)
plt.title("Newton Raphson Method    Sumoon[45]")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(root,0,color="red")
plt.show()

# def f(x):
#     return x**4 - x - 10
# def g(x):
#     return 4*x**3 - 1

# E=0.0005
# N=4
# x0=int(input("Input initial guess: "))

# i=1
# Hello=True

# while Hello:
#     if (g(x0))==0:
#         print("Mathematical Error")
#         break
        
#     x1= x0 - (f(x0)/g(x0))
#     x0=x1
#     i=i+1
#     if i>=N:
#         print("Not Convergence")
    
#     if (abs(f(x1)-f(x0))<E):
#         Hello=False

# print(f"The root is {x1}")                
        
        