
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - x - 3
    

def Secant(F, a, b):
  tol = 0.00005
  maxitr=100

  if abs(F(a))< tol:
    return round(a,5)

  if abs(F(b))< tol:
    return round(b,5)

  for i in range(maxitr):
    if abs(F(b)-F(a))<tol:
      print("Error! dividion by zero")
      return None

    c= (a*F(b)-b*F(a))/(F(b)-F(a))

    if abs(F(c))<tol:
      return round(c,5)

    a,b = b,c
  return None


root = Secant(f, 4, 5)
print("THE ROOT IS =", root)

X = np.linspace(-10, 10, 500)
Y = f(X)
plt.figure(figsize=(9, 2.5))
plt.plot(X, Y)
plt.grid()
plt.scatter(root, 0, color="red")
plt.show()






# import sys

# def f(x):
#     return x**3 - 2*x -5

# E=0.0005
# N=6
# print("Enter initial guesses: ")
# x1=int((input("Choose lower guess: ")))
# x2=int((input("Choose upper guess: ")))

# i=1
# Hello=True

# while Hello:
#     if (f(x1)==f(x2)):
#         print("Mathematical Error")
#         sys.exit()
    
#     x0= x2 - ((x2-x1) * f(x2))/ (f(x2)-f(x1))
#     x1=x2
#     x2=x0
#     i+=1
#     if i>=N:
#         print("Not Convergent ")
#         sys.exit()
#     if (f(x2)<E):
#          Hello=False
         
# print(f"The Root is {x2}")            


# import sys

# def f(x):
#     return x**3 - 2*x -5

# E=0.0005
# N=6
# print("Enter initial guesses: ")
# x1=int((input("Choose lower guess: ")))
# x2=int((input("Choose upper guess: ")))

# i=1
# Hello=True

# while Hello:
#     if (f(x1)==f(x2)):
#         print("Mathematical Error")
#         sys.exit()
    
#     x0= x2 - ((x2-x1) * f(x2))/ (f(x2)-f(x1))
#     x1=x2
#     x2=x0
#     i+=1
#     if i>=N:
#         print("Not Convergent ")
#         sys.exit()
#     if (f(x2)<E):
#          Hello=False
         
# print(f"The Root is {x2}")            
