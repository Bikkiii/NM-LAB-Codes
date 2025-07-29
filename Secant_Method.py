import sys

def f(x):
    return x**3 - 2*x -5

E=0.0005
N=6
print("Enter initial guesses: ")
x1=int((input("Choose lower guess: ")))
x2=int((input("Choose upper guess: ")))

i=1
Hello=True

while Hello:
    if (f(x1)==f(x2)):
        print("Mathematical Error")
        sys.exit()
    
    x0= x2 - ((x2-x1) * f(x2))/ (f(x2)-f(x1))
    x1=x2
    x2=x0
    i+=1
    if i>=N:
        print("Not Convergent ")
        sys.exit()
    if (f(x2)<E):
         Hello=False
         
print(f"The Root is {x2}")            
