import numpy as np
import matplotlib.pyplot as plt

def steep(A,x,b,tol,max_iter):
    errs = []
    err = tol + 1
    r = b - A @ x
    i = 0
    while err > tol and i < max_iter:
        alpha = (r @ r) / (r @ (A @ r))
        x += alpha * r
        r = b - A @ x
        err = np.linalg.norm(r)
        i += 1
        errs.append(err)

    if i == max_iter:
        print("Out of Iterations")

    return x, errs

def cg(A,x,b,tol,max_iter):
    errs = []
    err = tol + 1
    r = b - A @ x
    p = r
    i = 0
    while err > tol and i < max_iter:
        q = A @ p
        alpha = (r @ r) / (p @ q)
        x += alpha * p
        r_new = r - alpha * (A @ p)
        beta = (r_new @ r_new) / (r @ r)
        p = r_new + beta * p
        err = np.linalg.norm(r_new)
        r = r_new
        i += 1
        errs.append(err)

    if i == max_iter:
        print("Out of Iterations")

    return x, errs

def main():
    n_vals = [10, 100, 1000]
    colors = plt.get_cmap('tab10').colors

    fig = plt.figure(figsize=(18,10))
    fig.suptitle("Steepest Descent")
    ax = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)
    ax.set_title("n = 10")
    ax2.set_title("n = 100")
    ax3.set_title("n = 1000")
    ax.set_xlabel("Iteration Number")
    ax2.set_xlabel("Iteration Number")
    ax3.set_xlabel("Iteration Number")
    ax.set_ylabel("Error")
    ax2.set_ylabel("Error")
    ax3.set_ylabel("Error")

    fig2 = plt.figure(figsize=(18,10))
    fig2.suptitle("Conjugate Gradient Method")
    ax_2 = fig2.add_subplot(1,3,1)
    ax2_2 = fig2.add_subplot(1,3,2)
    ax3_2 = fig2.add_subplot(1,3,3)
    ax_2.set_title("n = 10")
    ax2_2.set_title("n = 100")
    ax3_2.set_title("n = 1000")
    ax_2.set_xlabel("Iteration Number")
    ax2_2.set_xlabel("Iteration Number")
    ax3_2.set_xlabel("Iteration Number")
    ax_2.set_ylabel("Error")
    ax2_2.set_ylabel("Error")
    ax3_2.set_ylabel("Error")

    fig3 = plt.figure(figsize=(18,10))
    fig3.suptitle("Assymetric Conjugate Gradient Method")
    ax_3 = fig3.add_subplot(1,3,1)
    ax2_3 = fig3.add_subplot(1,3,2)
    ax3_3 = fig3.add_subplot(1,3,3)
    ax_3.set_title("n = 10")
    ax2_3.set_title("n = 100")
    ax3_3.set_title("n = 1000")
    ax_3.set_xlabel("Iteration Number")
    ax2_3.set_xlabel("Iteration Number")
    ax3_3.set_xlabel("Iteration Number")
    ax_3.set_ylabel("Error")
    ax2_3.set_ylabel("Error")
    ax3_3.set_ylabel("Error")

    for n in n_vals:
        k_vals = np.linspace(1,10,10)
        i = 0
        for k in k_vals:
            print(i)
            Q, R = np.linalg.qr(np.random.randn(n,n))
            D = np.diag(np.linspace(1,k,n))
            A = Q @ D @ np.transpose(Q)
            b = np.array(np.random.randn(n))
            b = b / np.linalg.norm(b)
            x = np.array(np.random.randn(n))
            x2 = np.copy(x)
            x3 = np.copy(x)

            tol = 1e-10
            max_iter = 2000

            # Steepest, solve and plot
            sol_s, errs = steep(A,x,b,tol,max_iter)
            iters = np.arange(0,len(errs),1)

            if n == 10:
                ax.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")
            elif n == 100:
                ax2.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")
            else:
                ax3.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")

            # CG, solve and plot
            sol_cg, errs = cg(A,x2,b,tol,max_iter)
            iters = np.arange(0,len(errs),1)
            if n == 10:
                ax_2.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")
            elif n == 100:
                ax2_2.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")
            else:
                ax3_2.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")

            epsilon = 0.01
            A += epsilon * np.random.randn(n,n)

            # CG, solve and plot
            max_iter = 100
            sol_cg, errs = cg(A,x3,b,tol,max_iter)
            iters = np.arange(0,len(errs),1)
            if n == 10:
                ax_3.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")
            elif n == 100:
                ax2_3.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")
            else:
                ax3_3.semilogy(iters,errs,color=colors[i],linestyle='-',label=f"k = {k}")


            i += 1

    ax.set_ylim(tol,1)
    ax2.set_ylim(tol,1)
    ax3.set_ylim(tol,1)
    ax.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")
    fig.savefig("steep_result.png")

    ax_2.set_ylim(tol,1)
    ax2_2.set_ylim(tol,1)
    ax3_2.set_ylim(tol,1)
    ax_2.legend(loc="upper right")
    ax2_2.legend(loc="upper right")
    ax3_2.legend(loc="upper right")
    fig2.savefig("cg_result.png")

    ax_3.set_ylim(tol,1)
    ax2_3.set_ylim(tol,1)
    ax3_3.set_ylim(tol,1)
    ax_3.legend(loc="upper right")
    ax2_3.legend(loc="upper right")
    ax3_3.legend(loc="upper right")
    fig3.savefig("assymetric_result.png")

main()