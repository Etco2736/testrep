import numpy as np

def build_hilbert(n):
    arr = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            arr[i,j] = 1/(i+j+1)

    return arr

def power_method(A, Nmax, tol):
    n = A.shape[0]
    b_k = np.random.rand(n)
    prev = 1000

    for i in range(Nmax):
        b_k1 = np.dot(A, b_k)
        b_k = b_k1 / np.linalg.norm(b_k1)
        lam = np.dot(b_k.T, np.dot(A, b_k))
        err = np.abs(lam - prev)
        prev = lam
        if err < tol:
            return b_k, lam, i

    lam = np.dot(b_k.T, np.dot(A, b_k))
    return b_k, lam, Nmax

def inverse_power_method(A, Nmax, tol):
    n = A.shape[0]
    b_k = np.random.rand(n)
    prev = 1000

    for i in range(Nmax):
        b_k1 = np.linalg.solve(A, b_k)
        b_k = b_k1 / np.linalg.norm(b_k1)
        lam = np.dot(b_k.T, np.dot(A, b_k))
        
        err = np.abs(lam - prev)
        prev = lam
        if err < tol:
            return b_k, lam, i

    lam = np.dot(b_k.T, np.dot(A, b_k))
    return b_k, lam, Nmax

def main():
    # Question 3
    Nmax = 10000
    tol = 1e-10

    n_vals = np.arange(4,24,4)
    print("Dominant Eigenvalues...")
    for n in n_vals:
        A = build_hilbert(n)
        v, lam, iter = power_method(A,Nmax,tol)
        print(f"n: {n}, lamda: {lam}, iterations: {iter}")

    print("Smallest Eigenvalues...")
    for n in n_vals:
        A = build_hilbert(n)
        v, lam, iter = inverse_power_method(A,Nmax,tol)
        print(f"n: {n}, lamda: {lam}, iterations: {iter}")

    # Breaking the solver.... (Part D)
    A = np.array([
        [1, 1],
        [0, 1]
    ])
    v, lam, iter = power_method(A,Nmax,tol)
    print(f"lamda: {lam}, iterations: {iter}")
    

main()