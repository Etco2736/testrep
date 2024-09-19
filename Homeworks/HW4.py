import numpy as np
import matplotlib.pyplot as plt
import scipy.special as s

def f1(x):
    return (35)*s.erf(x/2/np.sqrt(0.138e-6*(60*60*24*60)))+-15

def f1_prime(x):
    return (35)*2/np.sqrt(np.pi)*np.exp(-(60*60*24*60)**2)

def bisection(f,a,b,tol):
    fa = f(a)
    fb = f(b)
    err = 1e5
    count = 0
    if fa*fb > 0:
        print("Poor boundaries")
        return -1
    elif fa == 0:
        return a
    elif fb == 0:
        return b
    while err > tol:
        c = (a+b)/2
        fc = f(c)
        if fc*fb < 0:
            fa = fc
            a = c
        elif fa*fb < 0:
            fb = fc
            b = c   
        else:  
            print("No convergence") 
            return -1 
        err = abs((a-b)/2)
        count = count + 1
    print("Error:", err)
    print("Iterations:",count)
    return c

def newton(f,fp,x_0,tol):
    n_max = 500
    i = 0
    err = 500
    if fp(x_0) == 0:
        return 1
    while err > tol:
        x_1 = x_0-f(x_0)/fp(x_0)
        err = abs(x_1-x_0)
        x_0 = x_1
        if fp(x_0) == 0:
            return 1
        if i > n_max:
            return -1
        i = i + 1
    return x_0

def main():
    # Question 1
    # Part A
    x_bar = 2
    x = np.linspace(0,x_bar,30)
    y = f1(x)
    plt.plot(x,y,'-r')
    plt.xlabel("Depth (m)")
    plt.ylabel("Temperature (C)")
    # Part B
    tol = 10**(-13)
    sol1 = bisection(f1,0,x_bar,tol)
    # Part C
    sol2 = newton(f1,f1_prime,0.01,10**(-13))
    sol3 = newton(f1,f1_prime,x_bar,10**(-13))
    print(sol2,sol3)

    plt.show()


main()