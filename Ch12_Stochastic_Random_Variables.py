# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:06:20 2020

@author: Cipriandan
------------------------------------------------------------------------------


------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from pylab import plt, mpl
import scipy.stats as scs

import math

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'tahoma'



def print_statistics(a1, a2, a1_type, a2_type):
        ''' Prints selected statistics.        
        Parameters
        ==========
        a1, a2: ndarray objects
            results objects from simulation
        '''
        sta1 = scs.describe(a1)  
        sta2 = scs.describe(a2)
        print('%14s %14s %14s' % ('statistic', 'data set 1', 'data set 2'))
        print(45 * "-")
        print('%14s %14.0f %14.0f' % ('size', sta1[ 0 ], sta2[ 0 ]))
        print('%14s %14.3f %14.3f' % ('min', sta1[ 1 ][ 0 ], sta2[ 1 ][ 0 ]))
        print('%14s %14.3f %14.3f' % ('max', sta1[1][1], sta2[1][1]))
        print('%14s %14.3f %14.3f' % ('mean', sta1[2], sta2[2]))
        print('%14s %14.3f %14.3f' % ('std', np.sqrt(sta1[3]), np.sqrt(sta2[3])))
        print('%14s %14.3f %14.3f' % ('skew', sta1[4], sta2[4]))
        print('%14s %14.3f %14.3f' % ('kurtosis', sta1[5], sta2[5]))
        a1_sort = np.sort( a1 )
        a2_sort = np.sort( a2 )

        plt.scatter( x = a1_sort, y = a2_sort, marker = '.', color = 'darkred' )
        plt.plot( a1_sort, a1_sort, linestyle = 'dashed', color = 'darkblue', alpha = 0.4 )
        plt.xlabel( a1_type )
        plt.ylabel( a2_type )


#..................................
#  exp(  Standard Normal )  
#..................................
S0 = 100 
r = 0.05
sigma = 0.25
T = 2.0
N = 10 ** 4

expr = ( r - sigma**2 / 2 ) * T + sigma * np.sqrt( T ) * np.random.standard_normal( N )
ST1 = S0 * np.exp( expr )
del expr

plt.figure( figsize = (10, 6 ) )
plt.hist( ST1, bins = 100, edgecolor = 'darkgray', color = 'darkblue' )
plt.xlabel( 'index level' )
plt.ylabel( 'frequency' )
plt.title('Standard Normal')

#..................................
#  log-normal(  mu, sigma )  
#..................................
mu = ( r - sigma**2 / 2 ) * T
sigma_l = sigma * math.sqrt( T )

ST2 = S0 * np.random.lognormal( mu, sigma_l, size = N )

plt.figure( figsize = (10, 6 ) )
plt.hist( ST2, bins = 100, edgecolor = 'darkgray', color = 'darkred' )
plt.xlabel( 'index level' )
plt.ylabel( 'frequency' )
plt.title('Log-Normal')

#................ Compare the stats and scatter plot ..........
print_statistics( ST1, ST2, 'Standard Normal', 'Log-Normal' )



