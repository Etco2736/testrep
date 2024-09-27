import numpy as np
import matplotlib.pyplot as plt
import scipy.special as s

def f1(x):
    return (35)*s.erf(x/2/np.sqrt(0.138e-6*(60*60*24*60)))+-15

def f1_prime(x):
    return (35)/np.sqrt(np.pi*0.138e-6*60*60*24*60)*np.exp(-(x/2/np.sqrt(60*60*24*60*0.138e-6))**2)

def f4(x):
    return np.exp(3*x)-27*x**6+27*x**4*np.exp(x)-9*x**2*np.exp(2*x)

def f4_prime(x):
    return 3*(np.exp(x)-6*x)*(np.exp(x)-3*x**2)**2

def f4_2prime(x):
    return 9*(-90*x**4 + 3*np.exp(x)*(x**2+8*x+12)*x**2 - 2*np.exp(2*x)*(2*x**2+4*x+1)+ np.exp(3*x))

def f5(x):
    return x**6 - x - 1

def f5_prime(x):
    return 6*x**5 - 1

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
    return x_0,p,vals

def newton2(f,fp,x_0,m,tol):
    n_max = 500
    i = 0
    err = 500
    p = []
    vals = []
    if fp(x_0) == 0:
        return 1
    while err > tol:
        x_1 = x_0-m*f(x_0)/fp(x_0)
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
    return x_0,p,vals

def newton3(f,fp,fp2,x_0,tol):
    n_max = 500
    i = 0
    err = 500
    p = []
    vals = []
    if fp(x_0) == 0:
        return 1
    while err > tol:
        x_1 = x_0 - f(x_0)/(fp(x_0)-f(x_0)*fp2(x_0)/fp(x_0))
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
    return x_0,p,vals

def secant(f,a,b,tol):
    i = 0
    err = 500
    p = []
    vals = []
    while err > tol:
        c = b - f(b)*(b-a)/(f(b)-f(a))
        a = b
        b = c
        err = np.abs(c-a)
        i = i + 1
        if f(b)-f(a) == 0:
            return -1,p,vals
        p.append(err)
        vals.append(c)
    print("Iterations: ", i)
    return c,p,vals

def q4():
    x,p1,val1 = newton(f4,f4_prime,4,10**-13)
    y,p2,val2 = newton2(f4,f4_prime,4,2,10**-13)
    z,p3,val3 = newton3(f4,f4_prime,f4_2prime,4,10**-13)
    print(x,y,z)
    print(f4(x),f4_prime(x),f4_2prime(x))
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    v1 = np.arange(0,p1.shape[0],1)
    v2 = np.arange(0,p2.shape[0],1)
    v3 = np.arange(0,p3.shape[0],1)
    plt.plot(v1,p1,'-r',label=r"$g(x) = x-\frac{f(x)}{f'(x)}$")
    plt.plot(v2,p2,'-b',label=r"$g(x) = x-m\frac{f(x)}{f'(x)}$")
    plt.plot(v3,p3,'-g',label=r"$g(x) = x-\frac{f(x)}{f'(x)-\frac{f(x)f''(x)}{f'(x)}}$")
    plt.legend(loc = 'upper right')
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.show()
    plt.clf()
    v1 = np.array(val1[1:])
    v2 = np.array(val2[1:])
    v3 = np.array(val3[1:])
    val1 = np.array(val1[:-1])
    val2 = np.array(val2[:-1])
    val3 = np.array(val3[:-1])
    plt.loglog(np.abs(val1-x),np.abs(v1-x),'-r',label=r"$g(x) = x-\frac{f(x)}{f'(x)}$")
    plt.loglog(np.abs(val2-y),np.abs(v2-y),'-b',label=r"$g(x) = x-m\frac{f(x)}{f'(x)}$")
    plt.loglog(np.abs(val3-z),np.abs(v3-z),'-g',label=r"$g(x) = x-\frac{f(x)}{f'(x)-\frac{f(x)f''(x)}{f'(x)}}$")
    plt.xlabel(r"$|x_n - \alpha|$")
    plt.ylabel(r"$|x_{n+1} - \alpha|$")
    plt.legend(loc = 'lower right')
    plt.show()

def q5():
    x,p1,val1 = newton(f5,f5_prime,2,10**-13)
    y,p2,val2 = secant(f5,2,1,10**-13)
    p1 = np.array(p1)
    p2 = np.array(p2)
    v1 = np.arange(0,p1.shape[0],1)
    v2 = np.arange(0,p2.shape[0],1)
    plt.plot(v1,p1,'-r',label="Newton")
    plt.plot(v2,p2,'-b',label="Secant")
    plt.legend(loc = 'upper right')
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")
    plt.show()
    plt.clf()
    v1 = np.array(val1[1:])
    v2 = np.array(val2[1:])
    val1 = np.array(val1[:-1])
    val2 = np.array(val2[:-1])
    plt.loglog(np.abs(val1-x),np.abs(v1-x),'-r',label="Newton")
    plt.loglog(np.abs(val2-y),np.abs(v2-y),'-b',label="Secant")
    plt.xlabel(r"$|x_n - \alpha|$")
    plt.ylabel(r"$|x_{n+1} - \alpha|$")
    plt.legend(loc = 'lower right')
    plt.show()

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
    sol2,p,v = newton(f1,f1_prime,0.01,10**(-13))
    sol3,p,v = newton(f1,f1_prime,x_bar,10**(-13))
    print(sol1,sol2,sol3)
    plt.vlines(x=sol1, ymin = -15, ymax=15, linestyle='-', color='k')
    plt.hlines(y=0, xmin = 0, xmax = 2, linestyle='-',color='k')
    plt.show()

    # Question 4
    print("Question 5")
    q4()

    # Question 5
    print("Question 5:")
    q5()
 

main()