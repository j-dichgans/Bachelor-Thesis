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


#in this version we compute var on the filtered time series (to catch more dynamic in f) and ar on the "unfiltered" to have memory effect preserved

#define time interval, T should be integer
T = 5000

def mu(t):
    return 2/T * t - 1

def f(x,t): 
    return (x - x**3 /3.0 - mu(t))



# #get equilibria curves, n should be T+1

def get_equilibria_paths():
    n = T + 1
    ts = np.linspace(0,T,n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    unstable = np.zeros(n)

    for i in tqdm.trange(n):
        sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,50),[2.0], method="BDF")
        upper[i] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x: f(x,ts[i]),(0.0,50),[-2.0], method="BDF")
        lower[i] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x: -f(x,ts[i]),(0.0,50),[0.0], method="BDF")
        unstable[i] = sol.y[0,-1]
     
    unstable[np.abs(unstable)>1] = np.nan

    return ts,upper,lower,unstable


#numerical simulation of SDEs:

steps_per_unit_time = 10                                             
solve_ts = np.linspace(0,T,T*steps_per_unit_time + 1)
dt = 1/steps_per_unit_time



#white noise case

xs_white = np.zeros(T*steps_per_unit_time + 1)
xs_white[0] = 2.103803356099623

sigma = 0.05
white_noise = np.random.normal(0,np.sqrt(dt),T*steps_per_unit_time)


for i in tqdm.trange(T*steps_per_unit_time):
    xs_white[i+1] = xs_white[i] + f(xs_white[i],solve_ts[i])*dt + sigma*white_noise[i]



#red noise case

xs_red = np.zeros(T*steps_per_unit_time+1)
xs_red[0] = 2.103803356099623                                                       #upper[0]
kappa = 0.1                                                                        #shouldn't be too small 

def eta(theta):
    eta = np.zeros(T*steps_per_unit_time)
    for i in tqdm.trange(T*steps_per_unit_time - 1):
        eta[i+1] = np.exp(-theta*dt)*eta[i] + np.sqrt(1/(2*theta)*(1-np.exp(-2*theta*dt)))*np.random.normal(0,1)
    return eta

eta = eta(1)


for i in tqdm.trange(T*steps_per_unit_time):
    xs_red[i+1] = xs_red[i] + f(xs_red[i],solve_ts[i])*dt + kappa*eta[i]*dt



xs_white_filtered = xs_white[::steps_per_unit_time]
xs_red_filtered = xs_red[::steps_per_unit_time]

window_length = 50 

def get_var(x):
    #get var of T-windows with length window length. Gives n_windows values
    n_windows = int(T/window_length) 
    var = np.full(n_windows,np.nan)
    for i in tqdm.trange(n_windows):
        var[i] = statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2).var()
    return var
        
        

def get_ar(x):
    #x should have size T*steps_per_unit_time + 1
    ar = np.full(T+1,np.nan)
    for i in tqdm.tgrange(T):
        ar[i+1] = statsmodels.tsa.stattools.acf(statsmodels.tsa.tsatools.detrend(x[i*steps_per_unit_time:(i+1)*steps_per_unit_time],order=2))[1]
    return ar


#ToDo: deal with size(eta) + 1 = size xs, should be fine?
#vorher: (1/dt)**2, in Aufruf: dt = dt/epsilon, und wieso fs = 1/dt

def get_lambdas(x,noise,method="ratio"):
    
    def fitfunction(f,ls):
        return 1/((2*np.pi*f)**2  + ls**2)          #omega = 2*pi*f ? why not 1 in nominator # vorher: (1/dt)**2
    
    
    n_windows = int(T/window_length) 
    ls = np.full(n_windows,np.nan)
    
    for i in tqdm.trange(n_windows):
        detx = statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2)
        f,Sxx = scipy.signal.welch(detx)                                    #was ist hier der frequency domain?
        f,Sff = scipy.signal.welch(noise[i*window_length:(i+1)*window_length])  #,detrend="linear")
            
        if method == "ratio":
            popt = scipy.optimize.curve_fit(fitfunction,
                                                  f[1:], 
                                                  Sxx[1:]/Sff[1:],
                                                  p0=[1.0],
                                                  bounds=(0.0, np.inf))[0]
            ls[i+1] = popt[0]
    
    return ls



variance_series_white = get_var(xs_white_filtered)
variance_series_red = get_var(xs_red_filtered)
ac_series_white = get_ar(xs_white)
ac_series_red = get_ar(xs_red)

# ls_series_white = get_lambdas(xs_white_filtered,white_noise[::window_length])
# ls_series_red = get_lambdas(xs_red_filtered,eta[::window_length])


log_ac_white = np.log(ac_series_white)                           
log_ac_red = np.log(ac_series_red)


ts, upper, lower, unstable = get_equilibria_paths()

tipp_upper = np.argmin(upper>1)
tipp_lower = np.argmin(lower>-1)


tip_white = np.argmin(xs_white_filtered>1.0)
tip_red = np.argmin(xs_red_filtered>1.0)  


fig,axs = plt.subplots(nrows=4,ncols=1,sharex=True,figsize=(8,10)) 

axs[0].plot(ts[:tipp_upper],upper[:tipp_upper],color = "black")
axs[0].plot(ts[tipp_lower:],lower[tipp_lower:],color = "black")
axs[0].plot(ts[~np.isnan(unstable)],unstable[~np.isnan(unstable)],linestyle = "--")


axs[1].plot(ts[:tip_white],log_ac_white[:tip_white],color="blue")                               
ax1_var = axs[1].twinx()
ax1_var.plot(ts[window_length:tip_white:window_length],variance_series_white[:int(tip_white/window_length)],color="blue",linestyle="--")


axs[2].plot(ts[:tip_red],log_ac_red[:tip_red],color="red")
ax2_var = axs[2].twinx()
ax2_var.plot(ts[window_length:tip_red:window_length],variance_series_red[:int(tip_red/window_length)],color="red",linestyle="--")


axs[0].set_ylabel(r"$x$")
axs[1].set_ylabel(r"$\log AC$")
axs[2].set_ylabel(r"$\log AC$")
ax1_var.set_ylabel(r"Variance")
ax2_var.set_ylabel(r"Variance")


# axs[3].set_xlabel("t")    
# axs[3].plot(ts[window_length:tip_white:window_length],-ls_series_white[:int(tip_white/window_length)],color="blue")
# axs[3].plot(ts[window_length:tip_red:window_length],-ls_series_red[:int(tip_red/window_length)],color="red")
# # axs[3].plot(mus,-true_lambda,color="black")
# axs[3].set_ylabel(r"$\lambda$")

ax_0 = axs[0].twiny()
ax_0.set_xlim(-1,1)
ax_0.set_xlabel("mu")

plt.show()






