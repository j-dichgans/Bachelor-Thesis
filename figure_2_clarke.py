
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:48:59 2022

@author: jc1147
"""
import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
import matplotlib.colors 
import tqdm
import scipy.signal
import statsmodels.tsa.stattools
import statsmodels.api as sm
import string

pcolors = ["blue","red"]
linestyles = ["solid","dashed","dotted"]

plt.rcParams["figure.figsize"] = (9,6)
plt.rcParams["font.family"] = "serif"

def f_auto(x,t,mu):
    return x - x**3 /3.0 - mu
def f(x,t,epsilon):
    return f_auto(x,t,t)/epsilon

def find_equilibria(mus):
    forward_stable = np.zeros_like(mus)
    backward_stable = np.zeros_like(forward_stable)
    forward_unstable = np.zeros_like(forward_stable)
    for idx in tqdm.trange(mus.size):
        mu=mus[idx]
        sol = scipy.integrate.solve_ivp(lambda t,x:f_auto(x,t,mu),(0.0,100),[2.0],method="BDF")
        forward_stable[idx] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x:f_auto(x,t,mu),(0.0,100),[-2.0],method="BDF")
        backward_stable[idx] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x:f_auto(x,t,mu),(100.0,0),[0.0],method="BDF")
        forward_unstable[idx] = sol.y[0,-1]
    forward_unstable[np.abs(forward_unstable) > 100.0] = np.nan
    return forward_stable,backward_stable,forward_unstable

def window_var(x,length):
    var = np.full_like(x,np.nan)
    for i in tqdm.trange(var.size-length):
        var[i+length] = statsmodels.tsa.tsatools.detrend(x[i:i+length],order=2).var()
        #var[i+length] = scipy.signal.detrend(x[i:i+length]).var()
    return var

def window_ar(x,length):
    ar = np.full_like(x,np.nan)
    conf  = np.full((x.size,2),np.nan)
    for i in tqdm.trange(ar.size-length):
        ret = statsmodels.tsa.stattools.acf(statsmodels.tsa.tsatools.detrend(x[i:i+length],order=2),
                                            alpha=0.05,fft=True)
        #ret = statsmodels.tsa.stattools.acf(scipy.signal.detrend(x[i:i+length]),
        #                                    alpha=0.05,fft=True)
        ar[i+length] = ret[0][1]
        conf[i+length] = ret[1][1]
    return ar,conf

def window_spec(x,eta,length,dt,method="ratio"):
    assert x.shape == eta.shape
    assert length < x.size
    
    def fitfunction(f,ls):
        return (1/dt)**2/((2*np.pi*f)**2  + ls**2)
    
    
    ls = np.full_like(x,np.nan)
    ls_err = ls.copy()
    
    for i in tqdm.trange(ls.size-length):
        detx = statsmodels.tsa.tsatools.detrend(x[i:i+length],order=2)
        f,Sxx = scipy.signal.welch(detx,fs=1/dt)
        f,Sff = scipy.signal.welch(eta[i:i+length],fs=1/dt)#,detrend="linear")
            
        if method == "ratio":
            popt, pcov = scipy.optimize.curve_fit(fitfunction,
                                                  f[1:],
                                                  Sxx[1:]/Sff[1:],
                                                  p0=[1.0],
                                                  bounds=(0.0, np.inf))
            ls[i+length] = popt[0]
            ls_err[i+length] = pcov[0][0]
        else:
            def f2m(L):
                return Sff[1:] * (1/dt)**2/((2*np.pi*f[1:])**2  + L**2) - Sxx[1:]
            opt = scipy.optimize.least_squares(f2m,1.0,bounds=(0.0,np.inf))
            ls[i+length] = opt.x.item()
    return ls,ls_err


mus = np.linspace(-1,1,1000)
upper,lower,unstable = find_equilibria(mus)
upper[upper<0.0]=np.nan
lower[lower>0.0]=np.nan

jac = 1-upper**2
jac[np.isnan(upper)] = 1-lower[np.isnan(upper)]**2
true_lambda = np.abs(jac)
true_lambda[mus>2/3] = np.nan

epsilon = 0.01
Tstart = -1.0
Tend   = +1.0
npoints = 5000

dt = (Tend - Tstart)/npoints
xs = np.zeros(npoints)
ts = np.linspace(Tstart,Tend,npoints)
xs[0] = upper[0] 
xs_red = xs.copy()

np.random.seed(0)

W = np.random.normal(scale=np.sqrt(dt),size=xs.size)
W_red = np.zeros_like(W)

r=0.99

for i in range(W_red.size-1):
    W_red[i+1] = r * W_red[i] + np.sqrt((1-r**2))*W[i]

W_red *= 0.5

for i in tqdm.trange(npoints-1):
    xs[i+1] = xs[i] + f(xs[i],ts[i],epsilon) * dt + W[i]
    xs_red[i+1] = xs_red[i] + f(xs_red[i],ts[i],epsilon) * dt +W_red[i]

win = 500

ls_white,ls_err_white = window_spec(xs,W,win,dt/epsilon)
ls_red,ls_err_red = window_spec(xs_red,W_red,win,dt/epsilon)

tip_white = np.argmin(xs>1.0)
tip_red = np.argmin(xs>1.0)

np.random.seed(0)
xs_ns = xs.copy()
W = np.random.normal(scale=np.sqrt(dt),size=xs_ns.size)
W_ns = W.copy()
rs = np.linspace(0.2,0.7,xs_ns.size)
for i in range(W_ns.size-1):
    W_ns[i+1] = rs[i] * W_ns[i] + np.sqrt((1-rs[i]**2))*W[i]
    
for i in tqdm.trange(npoints-1):
    xs_ns[i+1] = xs_ns[i] + f(xs_ns[i],ts[0],epsilon) * dt + W_ns[i]

var_ns = window_var(xs_ns,win)
ar_ns,_ = window_ar(xs_ns,win)
ls_ns,_ = window_spec(xs_ns,W_ns,win,dt/epsilon)

fig,axs=plt.subplots(nrows=3,ncols=1,sharex=True,figsize=(8,10))
axs[0].plot(ts,xs_ns,color="black")
axs[1].plot(ts,np.log(ar_ns)/dt*epsilon,color="black")
axsx=axs[1].twinx()
axsx.plot(ts,var_ns,color="black",linestyle="--")
axs[2].plot(ts,ls_ns,color="black")
axs[2].axhline(true_lambda[0],color="black",linestyle="--")
for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axsx.spines["top"].set_visible(False)
axs[0].set_ylabel(r"$x$")
axs[1].set_ylabel(r"$\frac{1}{\Delta t} \log AC$")
axsx.set_ylabel(r"Variance")
axs[2].set_xlabel(r"$t$")
axs[2].set_ylabel(r"$\lambda$")
for n,a in enumerate(axs):
    a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=12, weight='bold')
plt.savefig("figure2.eps")
plt.show()

