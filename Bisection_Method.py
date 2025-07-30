import numpy as np
import matplotlib.pyplot as plt

def F(x):
  return x**3 - x - 3

def Bisection(F, a, b):
  tol = 0.00005
  maxitr = 100

  if abs(F(a))< tol:
    return round(a,5)

  if abs(F(b))< tol:
    return round(b,5)

  for i in range(maxitr):
    c = (a + b)/2

    if abs(F(c)) < tol:
      return round(c,5)

    if F(c) < 0:
      a = c

    if F(c) > 0:
      b = c

  return None


a = 1
b = 2
root = Bisection(F, a, b)
print("The root is ", root)

plt.figure(figsize=(5,5))
X = np.linspace(a,b,200)
Y = F(X)
plt.grid()
plt.plot(X,Y)
plt.scatter(root, 0, color = 'red')
plt.axhline(0,linestyle='--')
plt.show()



# def f(x):
#     return x * np.cos(x) - x**2 + 1
# E=0.0005


# print("Choose Initial Guesses: ")
# x0=int((input("Choose lower guess: ")))
# x1=int((input("Choose upper guess: ")))

# # Create x-values for function plotting
# x_vals = np.linspace(-5, 5, 500)
# y_vals = f(x_vals)

# # Setup plot
# plt.figure(figsize=(8, 6))
# plt.axhline(0, color='black', linestyle='--')  # x-axis
# plt.plot(x_vals, y_vals, label="f(x) = x³ - 4x - 9")  # function curve
# while (f(x0)*f(x1))>0:
#     print(f"f(x0) * f(x1) = {f(x0)*f(x1)}")
#     print("Invalid Input!!\nChoose Again")
#     x0=int((input("Choose lower guess: ")))
#     x1=int((input("Choose upper guess: ")))
# print(f"f(x0) * f(x1) = {f(x0)*f(x1)}")    
# hello = True
# while hello:
#     x2=(x0 + x1)/2
#     if (f(x0) * f(x2))<0:
#         x1=x2
#     else:
#         x0=x2
#     if abs(f(x2))<E:
#         hello=False
# print(f"The root is {x2}")         


# plt.title("Bisection Method ")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.legend()
# plt.scatter(x2,0,color='red',marker='x')
# plt.grid(True)
# plt.show()
