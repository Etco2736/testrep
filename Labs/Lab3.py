import numpy as np

def f_part_1(x):
    return x**2*(x-1)

def f_part_2a(x):
    return (x-1)*(x-3)*(x-5)

def f_part_2b(x):
    return (x-1)**2*(x-3)

def f_part_2c(x):
    return np.sin(x)

def bisection(f,a,b,tol):
    fa = f(a)
    fb = f(b)
    err = 1e5
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
        err = abs(f(c))
    print("Error:", err)
    return c

def f_part_3a(x):
    return x*(1+(7-x**5)/(x**2))**3

def f_part_3b(x):
    return x - (x**5-7)/(x**2)

def f_part_3c(x):
    return x - (x**5-7)/(5*x**4)

def f_part_3d(x):
    return x - (x**5-7)/(12)

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
    tol = 10**(-5)

    # Part 1
    print("1a:",bisection(f_part_1,0.5,2,tol))
    print("1b:",bisection(f_part_1,-1,0.5,tol))
    print("1c:",bisection(f_part_1,-1,2,tol))

    # Part 2
    print("2a:",bisection(f_part_2a,0,2.4,tol))
    print("2b:",bisection(f_part_2b,0,2,tol))
    print("2c:",bisection(f_part_2c,0,0.1,tol))
    print("2d:",bisection(f_part_2c,0.5,3*np.pi/4,tol))

    # Part 3
    tol = 10**(-10)
    print("3a:",fixed_point(f_part_3a,1,tol))
    print("3b:",fixed_point(f_part_3b,1,tol))
    print("3c:",fixed_point(f_part_3c,1,tol))
    print("3d:",fixed_point(f_part_3d,1,tol))
    print("For Comparison", 7**(1/5))

driver()


