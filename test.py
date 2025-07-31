# import numpy as np
# import matplotlib.pyplot as plt

# def f(x):
#     return x**3 - x -2

# def bisection(f,a,b):
    
#     E=0.00005
#     N=100
    
#     if abs(f(a))<E:
#         return round(a,5)
    
#     if abs(f(b))<E:
#         return round(b,5)
    
#     if f(a)*f(b)<0:
        
#         for i in range(N):
#             c=(a+b)/2
            
#             if abs(f(c))<E:
#                 return round(c,5)
            
#             if (f(a) * f(c))<0:
#                 b=c
#             else:
#                 a=c
            
#     return False

# root=bisection(f,1,3)   
# print(f"Root is {root}")       

# X=np.linspace(-8,8,500)      
# Y=f(X)
# plt.plot(X,Y)
# plt.grid(True,linestyle='--')
# plt.title("Bisection method")
# plt.xlabel("X")
# plt.ylabel("f(X)")
# plt.axhline(0,0,color='red')
# plt.axvline(root,color='black',linestyle='--')
# plt.scatter(root,0,color='red')
# plt.show()






import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x* np.sin(x) + np.cos(x)

def secant(f,a,b):
    
    E=0.00005
    N=100
    
    if abs(f(a))<E:
        return round(a,5)
    
    if abs(f(b))<E:
        return round(b,5)
    
    for i in range(N):
        
        if abs(f(b)-f(a))<E:
            print("Error!!, divide by zero")
            return 
        
        c= (a*f(b) - b*f(a))/(f(b) - f(a))
        
        if abs(f(c))<E:
            return round(c,5)
        
        a=b
        b=c
    return False

j=0
for i in range(0,60):
    if(f(i) * f(i+1))<0:
        j+=1
        if(j==8):
            print(i)
            break
            
print(j)            
root=secant(f,i,i+1)        
print(f"The root is {root}")
    
X=np.linspace(-100,100,500)    
Y=f(X)

plt.plot(X,Y)
plt.grid(True,linestyle='--')
plt.xlabel("X")
plt.ylabel("f(X)")
plt.scatter(root,0,color='red')
plt.axhline(0,color='black')
plt.axvline(root,color='black')
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# E=0.0005

# def f(x):
#     return x**3 - x -2

# def g(x):
#     return 3*x**2 -1

# def Newton(f,g,a):
    
#     N=100
    
#     if abs(f(a))<E:
#         return round(a,5)
    
#     if g(a)==0:
#         print("Error!!")
#         return 
    
#     for i in range(N):
        
#         c= a- ( f(a)/g(a) )
        
#         if abs(f(c))< E:
#             return c
        
#         a=c
#     return False 


# root=Newton(f,g,i)
# print(f"The root is {root:.5f}")  

# X=np.linspace(-10,10,1000)
# Y=f(X)
# plt.plot(X,Y)
# plt.grid(True,ls='--')
# plt.scatter(root,f(root),color='red',zorder = 4)
# plt.axhline(0,color='black')
# plt.axvline(root,0,color='pink')
# # plt.plot(root,0,'ro')
# plt.show()

