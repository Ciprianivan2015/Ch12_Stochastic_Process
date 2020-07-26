# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:10:36 2020

@author: Cipriandan
--------------------------------------------
Stochastic processes:
---------------------
- Roughly speaking, a stochastic process is a sequence of random variables  
- In that sense, one should expect something similar to a sequence of 
repeated simulations of a random variable when simulating a process.         

Geometric Brownian Motion
-------------------------
Stochastic differential equation
        - dSt = r*St * dt + sigma * St * dZt

Discretization using Euler scheme:
        - St = S(t-dt) * exp( (r - 0.5* sigma^2) * dt + sigma * sqrt( dt ) * zt )



To read about:
--------------
1. Discretization using Euler scheme
2. Cholesky decomposition of matrices         
--------------------------------------------
"""

import numpy as np
import pandas as pd
from pylab import plt, mpl
import scipy.stats as scs

import math

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'tahoma'

S0 = 100 
r = 0.05
sigma = 0.25
T = 2.0
N = 10 ** 4  # .... number of paths to be simulated .................
M = 100       # .... number of time intervals for discretisation .....
dt = T / M

S = np.zeros( (M + 1, N ) ) 
S[ 0 ] = S0

for t in range( 1, M + 1 ):
        expr = ( r - 0.5 * sigma **2  ) * dt + sigma * math.sqrt( dt ) * np.random.standard_normal( N )
        S[ t ] = S[ t - 1 ] * np.exp( expr )
        del expr
        
S_T = S[ -1 ]        


#............................................................................
#   Histogram of final values
#............................................................................
plt.hist( S_T, bins = 100, edgecolor = 'darkgray', color = 'darkblue' )
plt.axvline( S0, linestyle = 'dashed', alpha = 0.8, color = 'darkred' )
plt.xlabel('index level' )
plt.ylabel('frequency' )
plt.title( 'Geometric Brownian Motion: distribution of S_T' )


#............................................................................
#  Sample of paths
#............................................................................
plt.plot( S[:, :10], lw = 0.85, linestyle = '-.' )
plt.xlabel('time')
plt.xlabel('index level')



#----------------------------------------------------------------------------
#    Mean-reverting processes
#------------------------------
#   - square root diffusion: dxt = k*( theta - xt ) * dt + sigma * sqrt( xt ) * dZ
#         - values of xt == Chi squared
#         - Euler discretization is biased for NON ( geometric Brownian motion )
#         - but, more desirable for numerical reasons
#----------------------------------------------------------------------------

x0 = 0.05
kappa = 3.0
theta = 0.02
sigma = 0.1
T = 2.0
N = 10 ** 5
M = 200
dt = T / M

def srd_euler():
        xh = np.zeros( ( M + 1, N) )
        x = np.zeros_like( xh ) 
        xh[ 0 ] = x0
        x[ 0 ] = x0
        for t in range( 1, M + 1 ):
                xh[ t ] = ( xh[ t - 1 ] + kappa * ( theta - np.maximum( xh[ t-1], 0 ) ) * dt + sigma * np.sqrt(  np.maximum( xh[ t-1], 0  )  ) * math.sqrt( dt ) * np.random.standard_normal( N )  )
        x = np.maximum( xh,0 )
        return x


%time x1 = srd_euler()

#............................................................................
#   Histogram of final values
#............................................................................
plt.hist( x1[ -1 ], bins = 100, edgecolor = 'darkgray', color = 'darkred' )
plt.xlabel( 'value' )
plt.ylabel('frequency')
plt.axvline( theta, color = 'darkblue', alpha = 0.75, linestyle = 'dashed' )
plt.title( 'Mean reverting process: Euler discretization' )
                

#............................................................................
#  Sample of paths
#............................................................................
plt.plot( x1[:, :5], lw = 0.85, linestyle = '-.' )
plt.xlabel('time')
plt.xlabel('Level')
plt.axhline( theta, color = 'darkblue', alpha = 1.75, linestyle = 'dashed' )
plt.title( 'Mean reverting process: Euler discretization' )



#----------------------------------------------------------------------------
#    Stochastic volatility
#----------------------------
#   Model: Heston ( 1993 )
#---------------------------------------------------------------------------------------
#      dSt = r * St * dt + sqrt( vt ) * St * dZ1t 
#      dvt = kv * (theta - vt) * dt + sigma * sqrt( vt ) * dZ2t
#      dZ1 * dZ2 = rho ... (leverage effect: volatility increases in declining markets )
#---------------------------------------------------------------------------------------

S0 = 100
r = 0.05
v0 = 0.1
kappa = 3.0 
theta = 0.25
sigma = 0.1
rho = 0.6
T = 1.0

corr_mat = np.zeros( (2,2) )
corr_mat[ 0,:] = [ 1.0, rho ]
corr_mat[ 1,:] = [ rho, 1.0 ]
cho_mat = np.linalg.cholesky( corr_mat )


M = 50
N = 5 * 10 ** 4
dt = T / M

ran_num = np.random.standard_normal( ( 2, M + 1, N ) )


#....................................................................
#         Stochastic volatility process
#....................................................................
v = np.zeros_like( ran_num[ 0 ] )
vh= np.zeros_like( ran_num[ 0 ] )

v[ 0 ] = v0
vh[ 0 ] = v0
for t in range( 1, M + 1 ):
        ran = np.dot( cho_mat, ran_num[:, t, :] )
        vh[ t ] = ( vh[ t - 1 ] + kappa * ( theta - np.maximum( vh[t-1],0 ) ) * dt + sigma * np.sqrt( np.maximum( vh[ t-1],0 )) * math.sqrt( dt ) * ran[ 1 ] )
        
v = np.maximum( vh, 0 )   

#....................................................................
#         Stochastic "stock" process
#....................................................................
S = np.zeros_like( ran_num[ 0 ] )     
S[ 0 ] = S0
for t in range( 1, M + 1 ):
        ran = np.dot( cho_mat, ran_num[:, t, :])
        S[ t ] = S[ t - 1 ] * np.exp( (r - 0.5 * v[ t ]) *dt + np.sqrt( v[ t ] ) * ran[ 0 ] * np.sqrt( dt ) )
        
#....................................................................
#         Plots
#....................................................................        
fig, ax = plt.subplots( 2, 2, figsize = (10, 6) )        
ax[0,0].hist( S[ -1 ], bins = 50, edgecolor = 'darkgray', color = 'darkred', alpha = 0.6  )
ax[0,0].axvline( S0, color = 'darkblue', linestyle = 'dashed', lw = 1.75, alpha = 0.75 )
ax[0,0].set_xlabel( 'Stock level: $S_T$' )
ax[0,0].set_ylabel( 'Frequency' )
ax[0,0].set_title( 'Distribution of $S_T$' )


ax[0,1].hist( v[ -1 ], bins = 50, edgecolor = 'darkgray', color = 'darkgreen', alpha = 0.6 )
ax[0,1].axvline( theta, color = 'darkblue', linestyle = 'dashed', lw = 1.75, alpha = 0.75 )
ax[0,1].set_xlabel('Stochastic volatility: $\sigma_t$')
ax[0,1].set_title( 'Distribution of $\sigma_t$' )

ax[1,0].plot( S[:, :10], lw = 1.5 )
ax[1,0].axhline( S0, color = 'darkblue', linestyle = 'dashed', lw = 1.75, alpha = 0.75 )
ax[1,0].set_ylabel( 'Stock level: $S_t$' )


ax[1,1].plot( v[:, :10], lw = 1.5 )
ax[1,1].axhline( theta, color = 'darkblue', linestyle = 'dashed', lw = 1.75, alpha = 0.75 )
ax[1,1].set_ylabel( 'Stochastic volatility: $\sigma_t$' )








