def f(x):
    return x**4 - x - 10
def g(x):
    return 4*x**3 - 1

E=0.0005
N=4
x0=int(input("Input initial guess: "))

i=1
Hello=True

while Hello:
    if (g(x0))==0:
        print("Mathematical Error")
        break
        
    x1= x0 - (f(x0)/g(x0))
    x0=x1
    i=i+1
    if i>=N:
        print("Not Convergence")
    
    if (abs(f(x1)-f(x0))<E):
        Hello=False

print(f"The root is {x1}")                
        
        