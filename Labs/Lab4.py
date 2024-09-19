import numpy as np

# Excersize 2.2, 1
def compute_order(x,xstar):
    diff1 = np.abs(x[1::]-xstar)
    diff2 = np.abs(x[0:-1]-xstar)
    fit = np.polyfit(np.log(diff2.flatten()),np.log(diff1.flatten()),1)
    _lambda = np.exp(fit[1])
    alpha = fit[0]
    return alpha,_lambda

def f1(x):
    return np.sqrt(10/(x+4))

def fixed_point(f,x_0,tol):
    err = 10**5
    count = 0
    p = []
    p.append(x_0)
    while err > tol:
        x = f(x_0)
        err = abs(x - x_0)
        x_0 = x
        p.append(x_0)
        count = count + 1
        if count > 1e5 or abs(x) > 1e5 or abs(x_0) > 1e5:
            print("No Convergence")
            break
    print("Iterations:",count)
    p = p[:count]
    return x, p

def aitkens(x):
    xn = np.array(x[:-2])
    xn1 = np.array(x[1:-1])
    xn2 = np.array(x[2:])
    return xn - (xn1-xn)**2/(xn2-2*xn1+xn)

def steffenson(f,a,tol):
    err = 10**5
    count = 0
    p = []
    p.append(a)
    while err > tol:
        b = f(a)
        c = f(b)
        new = a - (b-a)**2/(c-2*b+a)
        err = abs(new - a)
        a = new
        p.append(a)
        count = count + 1
        if count > 1e5 or abs(a) > 1e5:
            print("No Convergence")
            break
    print("Iterations:",count)
    p = p[:count]
    return a, p

def driver():
    print("Excersize 2.2, 2")
    xstar, x = fixed_point(f1,1.5,10**(-10))
    print("Order:",compute_order(x,xstar))

    print("Excersize 3.2")
    # ??? Didn't finish this part...
    print(aitkens(x))

    print("Excersize 3.4")
    xstar, x = steffenson(f1,1.5,10**(-10))
    print("Order:",compute_order(x,xstar))
driver()