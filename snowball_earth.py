import numpy as np
import matplotlib.pyplot as plt
import tqdm
import scipy.integrate

Q_0 = 342
eps = 0.6
delt = 5.67 * 1e-8

def alpha(T):
    return 0.5 - 0.2*np.tanh((T-265)/10)

temp = np.linspace(220,310,500)

# plt.plot(temp,(1-alpha(temp))*Q_0)
# plt.plot(temp, eps*delt*(temp**4))
# plt.show()

# def Q_ramp(t):
#     Q_start = 0.8
#     Q_end = 1.3
#     return Q_end*t/Tend + Q_start*(1-t/Tend)

epsilon = 0.01

def f_auto(T,t,Q):                                             
    return ((1-alpha(T))*Q - eps*delt*(T**4))*100
def f(T,t):
    return ((1-alpha(T))*t - eps*delt*(T**4))/epsilon


def find_equilibria(qs):
    forward_stable = np.zeros_like(qs)
    for idx in tqdm.trange(qs.size):
        q =qs[idx]
        sol = scipy.integrate.solve_ivp(lambda t,T:f_auto(T,t,q*Q_0),(0.0,100),[290.0],method="BDF")
        forward_stable[idx] = sol.y[0,-1]
    return forward_stable


qs = np.linspace(0.8,0.9,1000)
upper = find_equilibria(qs)

# plt.plot(qs,upper)
# plt.show()

Tstart = 0.8
Tend   = 1.3
npoints = 5000

dt = (Tend - Tstart)/npoints
Ts = np.zeros(npoints)
ts = np.linspace(Tstart,Tend,npoints)
Ts[0] = find_equilibria(np.array([Tstart]))

W = np.random.normal(scale=np.sqrt(dt),size=Ts.size)

for i in tqdm.trange(npoints-1):
    Ts[i+1] = Ts[i] + f(Ts[i],ts[i]*Q_0) * dt + W[i]*5

plt.plot(ts,Ts)
plt.show()







