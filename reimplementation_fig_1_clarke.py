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
T = 100

def mu(t):
    return 2/T * t - 1

epsilon = 0.01

def f(x,t): 
    return (x - x**3 /3.0 - mu(t))/epsilon



# #get equilibria curves, n should be T+1

def get_equilibria_paths(n):
    ts = np.linspace(0,T,n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    unstable = np.zeros(n)

    for i in tqdm.trange(n):
        sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,100),[2.0],method="BDF")
        upper[i] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,100),[-2.0],method="BDF")
        lower[i] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x: -f(x,ts[i]),(0.0,100),[0.0],method="BDF")
        unstable[i] = sol.y[0,-1]
     
    unstable[np.abs(unstable)>1] = np.nan

    return ts,upper,lower,unstable



#numerical simulation of SDEs:

steps_per_unit_time = 1000                                              #must be high enough!
ts = np.linspace(0,T,T*steps_per_unit_time + 1)
dt = 1/steps_per_unit_time



#white noise case

xs_white = np.zeros(T*steps_per_unit_time + 1)
xs_white[0] = 2.103803356099623

sigma = 1
white_noise = np.random.normal(0,np.sqrt(dt),T*steps_per_unit_time)

x_new = xs_white[0]

for i in tqdm.trange(T*steps_per_unit_time):
    xs_white[i+1] = xs_white[i] + f(xs_white[i],ts[i])*dt + sigma*white_noise[i]



#red noise case

xs_red = np.zeros(T*steps_per_unit_time+1)
xs_red[0] = 2.103803356099623                                                   #upper[0]
kappa = 10                                                                      #shouldn't be too small 

def eta(theta):
    eta = np.zeros(T*steps_per_unit_time)
    for i in tqdm.trange(T*steps_per_unit_time - 1):
        eta[i+1] = np.exp(-theta*dt)*eta[i] + np.sqrt(1/(2*theta)*(1-np.exp(-2*theta*dt)))*np.random.normal(0,1)
    return eta

eta = eta(1)


for i in tqdm.trange(T*steps_per_unit_time):
    xs_red[i+1] = xs_red[i] + f(xs_red[i],ts[i])*dt + kappa*eta[i]*dt



xs_white_filtered = xs_white[::steps_per_unit_time]
xs_red_filtered = xs_red[::steps_per_unit_time]


def get_var(x,T):
    #computes var of detrended time series for the T frames [0,1),...,[T-1,T)
    #x should have size T*steps_per_unit_time + 1   
    var = np.full(T+1,np.nan)
    for i in tqdm.trange(T):
        var[i+1] = statsmodels.tsa.tsatools.detrend(x[i*steps_per_unit_time:(i+1)*steps_per_unit_time],order=2).var()
    return var
        
        
    
def get_ar(x,T):
    #x should have size T*steps_per_unit_time + 1
    ar = np.full(T+1,np.nan)
    for i in tqdm.tgrange(T):
        ar[i+1] = statsmodels.tsa.stattools.acf(statsmodels.tsa.tsatools.detrend(x[i*steps_per_unit_time:(i+1)*steps_per_unit_time],order=2))[1]
    return ar


#ToDo: deal with size(eta) + 1 = size xs, should be fine?
#vorher: (1/dt)**2, in Aufruf: dt = dt/epsilon, und wieso fs = 1/dt

def get_lambdas(x,eta,dt,method="ratio"):
    
    def fitfunction(f,ls):
        return 1/((2*np.pi*f)**2  + ls**2)          #omega = 2*pi*f ? why not 1 in nominator # vorher: (1/dt)**2
    
    
    ls = np.full(T+1,np.nan)
    
    for i in tqdm.trange(T):
        detx = statsmodels.tsa.tsatools.detrend(x[i*steps_per_unit_time:(i+1)*steps_per_unit_time],order=2)
        f,Sxx = scipy.signal.welch(detx,fs=1/dt)                                    #was ist hier der frequency domain?
        f,Sff = scipy.signal.welch(eta[i*steps_per_unit_time:(i+1)*steps_per_unit_time],fs=1/dt)  #,detrend="linear")
            
        if method == "ratio":
            popt = scipy.optimize.curve_fit(fitfunction,
                                                  f[1:], 
                                                  Sxx[1:]/Sff[1:],
                                                  p0=[1.0],
                                                  bounds=(0.0, np.inf))[0]
            ls[i+1] = popt[0]
    
    return ls



variance_series_white = get_var(xs_white,T)
variance_series_red = get_var(xs_red,T)
ac_series_white = get_ar(xs_white,T)
ac_series_red = get_ar(xs_red,T)

ls_series_white = get_lambdas(xs_white,white_noise,dt)
ls_series_red = get_lambdas(xs_red,eta,dt)


log_ac_white = np.log(ac_series_white)/dt*epsilon                           #why epsilon, why dt?
log_ac_red = np.log(ac_series_red)/dt*epsilon


ts, upper, lower, unstable = get_equilibria_paths(T + 1)

tipp_upper = np.argmin(upper>1)
tipp_lower = np.argmin(lower>-1)


tip_white = np.argmin(xs_white_filtered>1.0)
tip_red = np.argmin(xs_red_filtered>1.0)  


fig,axs = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(8,10)) 

axs[0].plot(ts[:tipp_upper],upper[:tipp_upper],color = "black")
axs[0].plot(ts[tipp_lower:],lower[tipp_lower:],color = "black")
axs[0].plot(ts[~np.isnan(unstable)],unstable[~np.isnan(unstable)],linestyle = "--")
axs[0].plot(ts,xs_white_filtered,color="blue")
axs[0].plot(ts,xs_red_filtered,color="red")


axs[1].plot(ts[:tip_white],log_ac_white[:tip_white],color="blue")                               #np.nan in first place or [1:tip_white]?
ax1_var = axs[1].twinx()
ax1_var.plot(ts[1:tip_white],variance_series_white[1:tip_white],color="blue",linestyle="--")


axs[2].plot(ts[:tip_red],log_ac_red[:tip_red],color="red")
ax2_var = axs[2].twinx()
ax2_var.plot(ts[:tip_red],variance_series_red[:tip_red],color="red",linestyle="--")


axs[0].set_ylabel(r"$x$")
axs[1].set_ylabel(r"$\frac{1}{\Delta t} \log AC$")
axs[2].set_ylabel(r"$\frac{1}{\Delta t} \log AC$")
ax1_var.set_ylabel(r"Variance")
ax2_var.set_ylabel(r"Variance")


axs[3].set_xlabel("t")    
axs[3].plot(ts[:tip_white],-ls_series_white[:tip_white],color="blue")
axs[3].plot(ts[:tip_red],-ls_series_red[:tip_red],color="red")
# axs[3].plot(mus,-true_lambda,color="black")
axs[3].set_ylabel(r"$\lambda$")

ax_0 = axs[0].twiny()
ax_0.set_xlim(-1,1)
ax_0.set_xlabel("mu")

plt.show()

# axs[1].plot()


# plt.plot(np.arange(T+1),xs_white_filtered, c='blue')
# plt.plot(np.arange(T+1),xs_red_filtered, c='red')
# plt.show()





