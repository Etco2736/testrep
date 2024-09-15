import numpy as np
import matplotlib.pyplot as plt

def f_q1(x):
    return np.sin(x)+1-2*x

def f_q2a(x):
    return (x-5)**9

def f_q2b(x):
    return x**9-45*x**8+900*x**7-10500*x**6+78750*x**5-393750*x**4+1312500*x**3-2812500*x**2+3515625*x-1953125

def f_q3(x):
    return x**3+x-4

def f_q5a(x):
    return x-4*np.sin(2*x)-3

def f_q5b(x):
    return -np.sin(2*x)+5*x/4-3/4

def f_q5b_prime(x):
    return -2*np.cos(2*x)+5/4

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

def plot_5a(f,a,b):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.linspace(a,b,10000)
    ax.plot(x,f(x),'-r')
    ax.plot(x,np.zeros(10000),'-b')
    ax.plot(x,x,'-g')
    ax.set_aspect("equal")
    plt.show()

def plot_derivative(f,f2,a,b):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = np.linspace(a,b,10000)
    ax.plot(x,f(x),'-g',label="g'(x)")
    ax.plot(x,np.zeros(10000)-1,'--b')
    ax.plot(x,np.zeros(10000)+1,'--b')
    ax.plot(x,np.zeros(10000),'--m')
    ax.plot(x,f2(x),'-r',label="f(x)")
    ax.set_aspect("equal")
    ax.legend(loc='upper left')
    plt.show()

def fixed_point(f,x_0,tol):
    err = 10**5
    count = 0
    while err > tol:
        x = f(x_0)
        err = abs(x - x_0)
        x_0 = x
        count = count + 1
        if count > 1e5 or abs(x) > 1e5 or abs(x_0) > 1e5:
            print("No Convergence")
            break
    return x
    

def driver():
    # Question 1
    print("Question 1:")
    tol = 1e-8
    print(bisection(f_q1,0,np.pi,tol))

    # Question 2
    print("Question 2a")
    tol = 1e-4
    print(bisection(f_q2a,4.82,5.2,tol))
    print("Question 2b")
    print(bisection(f_q2b,4.82,5.2,tol))

    # Question 3
    print("Question 3")
    tol = 1e-3
    print(bisection(f_q3,1,4,tol))

    # Question 5
    print("Question 5:")
    plot_5a(f_q5a,-2.5,10)
    tol = 1e-11
    print(fixed_point(f_q5b,3,tol))
    plot_derivative(f_q5b_prime,f_q5a,-2.5,10)

driver()


