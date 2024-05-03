import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
import matplotlib.colors 
import tqdm
import scipy.signal
import statsmodels.tsa.stattools
import statsmodels.api as sm
import statsmodels.tsa.seasonal
import string

#define time interval, T should be integer
T = 400

def mu(t):
    return 2/T * t - 1

def f(x,t): 
    return x - x**3 /3.0 - mu(t)

# #get equilibria curves

# def get_equilibria_paths(n):
#     ts = np.linspace(0,T,n)
#     upper = np.zeros(n)
#     lower = np.zeros(n)
#     unstable = np.zeros(n)

#     for i in tqdm.trange(n):
#         sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,100),[2.0],method="BDF")
#         upper[i] = sol.y[0,-1]
#         sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,100),[-2.0],method="BDF")
#         lower[i] = sol.y[0,-1]
#         sol = scipy.integrate.solve_ivp(lambda t,x: -f(x,ts[i]),(0.0,100),[0.0],method="BDF")
#         unstable[i] = sol.y[0,-1]
     
#     unstable[np.abs(unstable)>1] = np.nan

#     return ts,upper,lower,unstable

# ts, upper, lower, unstable = get_equilibria_paths(T + 1)

# tipp_upper = np.argmin(upper>1)
# tipp_lower = np.argmin(lower>-1)

# plt.plot(ts[:tipp_upper],upper[:tipp_upper])
# plt.plot(ts[tipp_lower:],lower[tipp_lower:])
# plt.plot(ts[~np.isnan(unstable)],unstable[~np.isnan(unstable)])
# plt.show()

steps_per_unit_time = 1
ts = np.linspace(0,T,T*steps_per_unit_time + 1)
dt = 1/steps_per_unit_time

#red noise case

xs_red = np.zeros(T+1)
xs_red[0] = 2.103803356099623   #upper[0]
kappa = 1

def eta(theta):
    eta = np.zeros(T*steps_per_unit_time)
    for i in tqdm.trange(T*steps_per_unit_time - 1):
        eta[i+1] = np.exp(-theta*dt)*eta[i] + np.sqrt(1/(2*theta)*(1-np.exp(-2*theta*dt)))*np.random.normal(0,1)
    return eta

eta = eta(1)
x_new = xs_red[0]

for i in tqdm.trange(T):
    for j in tqdm.trange(steps_per_unit_time):
        x_new = x_new + f(x_new,ts[i*steps_per_unit_time + j])*dt + kappa*eta[i*steps_per_unit_time + j]*dt
    xs_red[i+1] = x_new


#white noise case

xs_white = np.zeros(T+1)
xs_white[0] = 2.103803356099623

sigma = 1
white_noise = np.random.normal(0,np.sqrt(dt),T*steps_per_unit_time)

x_new = xs_white[0]

for i in tqdm.trange(T):
    for j in tqdm.trange(steps_per_unit_time):
        x_new = x_new + f(x_new,ts[i*steps_per_unit_time + j])*dt + sigma*white_noise[i*steps_per_unit_time + j]
    xs_white[i+1] = x_new


#plotting

plt.plot(np.arange(T+1),xs_red,color = 'red')
plt.plot(np.arange(T+1),xs_white, color = 'blue')
plt.show()


# fig, ax1 = plt.subplots()
# ax1.plot(np.arange(T+1),xs_white)

# ax2 = ax1.twiny()
# ax2.plot(np.arange(T+1),mu(np.arange(T+1)))
# ax2.set_xlabel("mu")

# plt.show()








# steps_per_unit_time = 10
# n_steps = steps_per_unit_time*T
# dt = 1/steps_per_unit_time

# xs_white = np.zeros(T)
# xs_red = np.zeros(T)

# xs_white[0] = upper[0]
# xs_red[0] = upper[0]

# kappa = 1

# def eta(t):
#     eta = np.zeros(n_steps)




# for i in tqdm.trange(n_steps):
#     ts = np.linspace(0,T,n_steps+1)
#     xs_white[i+1] = xs_white[i] + f(xs_white[i],ts[i])*dt + 

