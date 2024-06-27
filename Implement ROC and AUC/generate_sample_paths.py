import numpy as np
import tqdm
import statsmodels.tsa.stattools
import scipy.signal
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

T = 14000
n_steps_per_unit_time = 10
dt = 1/n_steps_per_unit_time
window_length = 700
n_windows = int(T/window_length) 


n_parameter_settings = 50

np.random.seed(1)

l0s = np.random.uniform(0.3,0.5,n_parameter_settings)
th0s = np.random.uniform(0.5,4,n_parameter_settings)
thTs = np.random.uniform(0.5,4,n_parameter_settings)
k0s = np.random.uniform(0.5,4,n_parameter_settings)
kTs = np.random.uniform(0.5,4,n_parameter_settings)


#linear ramp for kappa
def kappa(t,k0,kT):
    return (1-t/T)*k0 + t/T*kT

#linear ramp for theta
def theta(t,th0,thT):
    return (1-t/T)*th0 + t/T*thT


def ls(t,l0):
    return l0*np.sqrt(1-t/T)



def get_var(x):
    #get var of T-windows with length window length. Gives n_windows values
    var = np.full(n_windows,np.nan)
    for i in tqdm.trange(n_windows):
        var[i] = statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2).var()
    return var


def get_ar(x):
#x should have size T*steps_per_unit_time + 1
    ar = np.full(n_windows,np.nan)
    for i in tqdm.trange(n_windows):
        ar[i] = statsmodels.tsa.stattools.acf(statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2))[1]
    return ar


def get_ls_k(x,noise):


    def fitfunction(f,ls):
        return np.log(1/(f**2  + ls**2))
    

    n_windows = int(T/window_length) 
    ls = np.full(n_windows,np.nan)



    for i in tqdm.trange(n_windows):
        frequencies = 2*np.pi*(1/window_length)*np.arange(1,window_length/2)                                                #here window_length should be even

        xs_window_detrend = statsmodels.tsa.tsatools.detrend(x[i*window_length:(i+1)*window_length],order=2)
        noise_window = noise[i*window_length:(i+1)*window_length]
        kappa_window = kappa(np.arange(i*window_length,(i+1)*window_length),k0,kT)                                                                               
        
        estim_psd_xs_wn = np.array([np.abs(1/np.sqrt(window_length)*(np.exp(-1j*frequencies[j]*np.arange(0,window_length)) @ xs_window_detrend))**2 for j in range(int(window_length/2) - 1)])
        estim_psd_xi_wn = np.array([np.abs(1/np.sqrt(window_length)*(np.exp(-1j*frequencies[n]*np.arange(0,window_length)) @ (noise_window*kappa_window)))**2 for n in range(int(window_length/2) - 1)])

        popt = scipy.optimize.curve_fit(fitfunction,
                                                frequencies, 
                                                np.log(estim_psd_xs_wn/estim_psd_xi_wn),
                                                p0=[1.0],
                                                bounds=(0.0, np.inf))[0]
        ls[i] = popt[0]

        
    return ls




def get_paths_decreasing_ls(l0,k0,kT,th0,thT):
    #simulate sample paths with euler method with decreasing ls 
    n_steps = T*n_steps_per_unit_time
    solve_ts = np.linspace(0,T,n_steps + 1)


    xs = np.zeros(n_steps+1)
    us = np.zeros(n_steps+1)

    #simulate us
    for i in tqdm.trange(n_steps):
        us[i+1] = np.exp(-theta(solve_ts[i],th0,thT)*dt)*us[i] + np.sqrt(1/(2*theta(solve_ts[i],th0,thT))*(1-np.exp(-2*theta(solve_ts[i],th0,thT)*dt)))*np.random.normal(0,1)

    for i in tqdm.trange(n_steps):
        xs[i+1] = xs[i] - ls(solve_ts[i],l0)*xs[i]*dt + kappa(solve_ts[i],k0,kT)*us[i]*dt

    xs_filtered = xs[::n_steps_per_unit_time]
    us_filtered = np.array([np.sum([us[i*n_steps_per_unit_time+j] for j in range(n_steps_per_unit_time)])*dt for i in range(T)])
    return xs_filtered, us_filtered



def get_paths_fix_l0(l0,k0,kT,th0,thT):
    #simulate sample paths with euler method with fix l0 value
    n_steps = T*n_steps_per_unit_time
    solve_ts = np.linspace(0,T,n_steps + 1)


    xs = np.zeros(n_steps+1)
    us = np.zeros(n_steps+1)

    #simulate us
    for i in tqdm.trange(n_steps):
        us[i+1] = np.exp(-theta(solve_ts[i],th0,thT)*dt)*us[i] + np.sqrt(1/(2*theta(solve_ts[i],th0,thT))*(1-np.exp(-2*theta(solve_ts[i],th0,thT)*dt)))*np.random.normal(0,1)

    for i in tqdm.trange(n_steps):
        xs[i+1] = xs[i] - l0*xs[i]*dt + kappa(solve_ts[i],k0,kT)*us[i]*dt

    xs_filtered = xs[::n_steps_per_unit_time]
    us_filtered = np.array([np.sum([us[i*n_steps_per_unit_time+j] for j in range(n_steps_per_unit_time)])*dt for i in range(T)])
    return xs_filtered, us_filtered



reference = np.arange(n_windows)

kendall_taus = np.zeros((n_parameter_settings*2,3))

for i in tqdm.trange(n_parameter_settings):
    l0, th0, thT, k0, kT = l0s[i], th0s[i], thTs[i], k0s[i], kTs[i]
    
    xs_filtered_decr_ls, us_filtered = get_paths_decreasing_ls(l0, th0, thT, k0, kT)
    xs_filtered_fix_l0, us_filtered = get_paths_fix_l0(l0, th0, thT, k0, kT)

    var_xs_decr_ls = get_var(xs_filtered_decr_ls)
    ar_xs_decr_ls = get_ar(xs_filtered_decr_ls)
    ls_xs_decr_ls = get_ls_k(xs_filtered_decr_ls,us_filtered)

    kendall_taus[2*i,0] = kendalltau(reference,var_xs_decr_ls)[0]
    kendall_taus[2*i,1] = kendalltau(reference,ar_xs_decr_ls)[0]
    kendall_taus[2*i,2] = kendalltau(reference,-ls_xs_decr_ls)[0]

    
    var_xs_fix_l0 = get_var(xs_filtered_fix_l0)
    ar_xs_fix_l0 = get_ar(xs_filtered_fix_l0)
    ls_xs_fix_l0 = get_ls_k(xs_filtered_fix_l0,us_filtered)

    
    kendall_taus[2*i+1,0] = kendalltau(reference,var_xs_fix_l0)[0]
    kendall_taus[2*i+1,1] = kendalltau(reference,ar_xs_fix_l0)[0]
    kendall_taus[2*i+1,2] = kendalltau(reference,-ls_xs_fix_l0)[0]


kendall_taus_decreasing_ls_var = kendall_taus[::2,0]
kendall_taus_decreasing_ls_ar = kendall_taus[::2,1]
kendall_taus_decreasing_ls_ls = kendall_taus[::2,2]


kendall_taus_fix_l0_var = kendall_taus[1::2,0]
kendall_taus_fix_l0_ar = kendall_taus[1::2,1]
kendall_taus_fix_l0_ls = kendall_taus[1::2,2]




#get ROC curve
n = 100
thresholds = np.linspace(1,-1,n)

true_positives = np.zeros((n,3))
false_positives = np.zeros((n,3))


for i in tqdm.trange(n):

    true_positives[i,0] = np.mean(kendall_taus_decreasing_ls_var > thresholds[i])
    true_positives[i,1] = np.mean(kendall_taus_decreasing_ls_ar >  thresholds[i])
    true_positives[i,2] = np.mean(kendall_taus_decreasing_ls_ls >  thresholds[i])

    false_positives[i,0] = np.mean(kendall_taus_fix_l0_var >  thresholds[i])
    false_positives[i,1] = np.mean(kendall_taus_fix_l0_ar >  thresholds[i])
    false_positives[i,2] = np.mean(kendall_taus_fix_l0_ls >  thresholds[i])


false_positives = false_positives*100
true_positives = true_positives*100


plt.plot(false_positives[:,0],true_positives[:,0], color = "green", label = 'Var')
plt.plot(false_positives[:,1],true_positives[:,1], color = "blue", label = 'AC(1)')
plt.plot(false_positives[:,2],true_positives[:,2], color = "orange", label = 'ROSA')

plt.xlabel('False positive rate [%]')
plt.ylabel('True positive rate [%]')
plt.title('Observed Data: 100%')

plt.grid(True)
plt.legend()

plt.show()






    
