
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import brute
import time
f = lambda x: x**4 - 3*x**2 + x
df = lambda x: 4*x**3 - 6*x + 1

def bis(f,df,a,b,tol):
    """
    """
    # Evaluate in the a, b
    if df(a) *df(b)>0:
        raise ValueError("The function does not have a root in the given interval")
    hs = [] # Iterations record
    iter = 1
    while True:
        wide = b - a
        c = (a + b)/2
        dfc = df(c)
        hs.append((iter,a,b,c,f(c),dfc,wide))
        text = f"""Iter{iter} \t a={a:.3f} \t b={b:.3f} \t c={c:.3f} \t f(c)={f(c):.3f} \t df(c)={dfc:.3f} \t wide={wide:.6f}"""
        print(text)
        if abs(dfc) < tol or wide < tol:
            return hs
        if df(a) * dfc > 0: # True df(b) < 0 therefore the root is in the interval [a,c]
            a = c
        else:
            b = c
        iter += 1
    

def plot():
    plt.figure()
    x = [100,625,2500,10e3]
    y_values = [-2.1131,-3.0645,-3.2827,-3.2873]
    y = lambda y: (-3.2769-y)/-3.2769
    error = [y(value) for value in y_values]
    plt.plot(x,error)
    plt.title("Error respecto a -3.2769 vs N")
    plt.grid()
    plt.xlabel("N")
    plt.ylabel("f(x,y)")
    plt.show()
    


def sci_brute(d):
    text = f"""--
GRID
--
d = {d}"""
    print(text)
    fb = lambda x : np.sum(x**2)
    r_ = tuple((-5,5) for _ in range(d))
    start = time.time()
    print (f"{brute(fb,r_,Ns=20)}")
    end = time.time()
    print(f"Time taken: {end - start}")
    return end - start

if __name__ == "__main__":
    a0, b0 = -2,2
    hs3 = bis(f,df,a0,b0,1e-3)
    bq3 = brentq(df,a0,b0,xtol=1e-3)
    error = (hs3[-1][3] - bq3) / bq3 * 100
    text = f"""--
ERROR RESPECTO A SCIPY: {error}%
--
"""
    print(text)
    hs6 = bis(f,df,a0,b0,1e-6)
    bq6 = brentq(df,a0,b0,xtol=1e-6)
    error = (hs6[-1][3] - bq6) / bq6 * 100
    text = f"""--
ERROR RESPECTO A SCIPY: {error}%
--
"""
    print(text)
    tol = 1e-9
    hs6 = bis(f,df,a0,b0,tol)
    bq6 = brentq(df,a0,b0,xtol=tol)
    error = (hs6[-1][3] - bq6) / bq6 * 100
    text = f"""--
ERROR RESPECTO A SCIPY: {error}%
--
"""
    
    print(text)
# %
    print("-----------------------------------------------")
    t1 = sci_brute(4)
    t2 = sci_brute(2)
    print(f"""---
    Diferencia de tiempo D=4 a D=2: {(t1-t2):.6f} s
    ---""")
    plot()


# %%
