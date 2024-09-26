import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(x**2+7*x-30)-1

def fp(x):
    return (2*x+7)*np.exp(x**2+7*x-30)

def f2p(x):
    return 2*np.exp(x**2+7*x-30)+(2*x+7)**2*np.exp(x**2+7*x-30)

def bisection_special(f,fp,f2p,a,b,tol):
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
        if np.abs(f(c)*f2p(c)/fp(c)**2) < 1:
            print("Iterations:",count)
            return newton(f,fp,c,tol)
    print("Error:", err)
    print("Iterations:",count)
    return c

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
    p = []
    vals = []
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
        p.append(err)
        vals.append(x_0)
    print('Iterations:', i)
    return x_0

def main():
    print(bisection(f,2,4.5,1e-10))
    print(newton(f,fp,4.5,1e-10))
    print(bisection_special(f,fp,f2p,2,4.5,1e-10))
main()
