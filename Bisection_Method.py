def f(x):
    return x**3 - 4*x -9 
E=0.0005


print("Choose Initial Guesses: ")
x0=int((input("Choose lower guess: ")))
x1=int((input("Choose upper guess: ")))
while (f(x0)*f(x1))>0:
    print(f"f(x0) * f(x1) = {f(x0)*f(x1)}")
    print("Invalid Input!!\nChoose Again")
    x0=int((input("Choose lower guess: ")))
    x1=int((input("Choose upper guess: ")))
print(f"f(x0) * f(x1) = {f(x0)*f(x1)}")    
hello = True
while hello:
    x2=(x0 + x1)/2
    if (f(x0) * f(x2))<0:
        x1=x2
    else:
        x0=x2
    if abs(f(x2))<E:
        hello=False
print(f"The root is {x2}")               