import numpy as np
import matplotlib.pyplot as plt
import math

def maclaurin(x):
    return x-x**3/6+x**5/math.factorial(5)-x**7/math.factorial(7)+x**9/math.factorial(9)-x**11/math.factorial(11)+x**13/math.factorial(13)
def pade33(x):
    return (x+(1/20-1/6)*x**3)/(1+x**2/20)
def pade24(x):
    return x/(1+x**2/6+(1/36-1/120)*x**4)

def main():
    x = np.linspace(0,5,10000)
    y = np.sin(x)
    y_taylor = maclaurin(x)
    y_pade1 = pade33(x)
    y_pade2 = pade24(x)
    err_taylor = y_taylor - y
    err_pade1 = y_pade1 - y
    err_pade2 = y_pade2 - y

    plt.title("Error Plot")
    plt.plot(x,err_taylor,'-r',label="Taylor")
    plt.plot(x,err_pade1,'-b',label="Part a,c")
    plt.plot(x,err_pade2,'-m',label="Part b")
    plt.legend(loc="lower left")
    plt.show()

main()