import numpy as np
import matplotlib.pyplot as plt

# 3.2 Excercises

# Question 1:
x = np.linspace(0,1,5)
y = np.arange(0,1,0.2)
# Question 2:
x2 = x[:3]
# Question 3:
print("The first three entries of x are:",x2)
# Question 4:
w = 10**(-np.linspace(1,10,10))
x = np.arange(1,w.size+1,1)
plt.semilogy(x,w)
plt.xlabel("x")
plt.ylabel("w")
plt.show()
# Question 5:
s = 3*w
plt.clf()
plt.semilogy(x,w,'-r',label="w")
plt.semilogy(x,s,'-b',label="s")
plt.xlabel("x")
plt.ylabel("Other Array")
plt.legend(loc="upper right")
plt.show()

# 4.2 Excersises
# Question 1
def driver():
    n = 2
    # New vectors which are orthagonal
    y = [0,1]
    w = [1,0]
    dp = dotProduct(y,w,n)
    print("the dot product is : ", dp)
    return   
def dotProduct(x,y,n):
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

# Question 2
def driver2():
    a = [[0,1],[1,0]]
    b = [[1,1],[1,1]]
    n = 2

    i = 0
    j = 0
    ans = [[0]*2]*2
    while i < n:
        while j < n:
            ans[i][j] = dotProduct(a[i],[b[0][j],b[1][j]],n)
            j = j + 1
        i = i + 1
    print(ans)
driver()
driver2()

# Question 3
a = np.array([[0,1],[1,0]])
b = np.array([[1,1],[1,1]])
print(np.matmul(a,b))
print(np.vdot([0,1],[1,0]))

'''
The numpy version is definitely faster I think
'''