import numpy as np
import tqdm
import scipy.integrate
import matplotlib.pyplot as plt

def f_auto(x,t,mu):                                             #why t?
    return x - x**3 /3.0 - mu
def f(x,t,epsilon):
    return f_auto(x,t,t)/epsilon


def find_equilibria(mus):
    forward_stable = np.zeros_like(mus)
    for idx in tqdm.trange(mus.size):
        mu=mus[idx]
        sol = scipy.integrate.solve_ivp(lambda t,x:f_auto(x,t,mu),(0.0,100),[2.0],method="BDF")
        forward_stable[idx] = sol.y[0,-1]
    return forward_stable


mus = np.linspace(-1,1,1000)
upper = find_equilibria(mus)

epsilon = 0.01
Tstart = -1.0
Tend   = +1.0
npoints = 5000

dt = (Tend - Tstart)/npoints
xs = np.zeros(npoints)
ts = np.linspace(Tstart,Tend,npoints)
xs[0] = upper[0] 

W = np.random.normal(scale=np.sqrt(dt),size=xs.size)

for i in tqdm.trange(npoints-1):
    xs[i+1] = xs[i] + f(xs[i],ts[i],epsilon) * dt + W[i]

plt.plot(ts,xs)
plt.plot(mus,upper)
plt.show()

