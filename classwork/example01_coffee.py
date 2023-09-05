#!/usr/bin/env python3

"""
Author: Lara Tobias-Tarsh
Created: 30/08/2023
Last Modified: 05/09/2023

Class coffee cooling problem. 
"""

import numpy as np
import matplotlib.pyplot as plt

#############
## GLOBALS ##
#############

tfinal, tstep = 600, 1
time_array = np.arange(0, tfinal, tstep)

###############
## FUNCTIONS ##
###############

def solve_temp(time, k=1/300., T_env=25, T_init=90):
    """
    Function takes an array of times and returns
    an array of temperatures corresponding to each time.

    Parameters
    -----------
    time : np.array
        array of time inputs for which you want temperatures
    k : int, optional
        proportionality constant of cooling
    T_env : int, optional
        integer representing the environmental temperature in celsius
    T_init : int, optional
        integer representing the initial temperature of the substance in celsius
    

    Returns
    --------
    temp : np.array
        array representing the temperature at each timestep of the cooling equation
    """
    
    temp = T_env + ( T_init - T_env ) * np.exp( -k * time )
    return temp


def time_to_temp(T_targ, k=1/300., T_env=20, T_init=90):
    """
    Given an initial temperature, an ambient temperature and a cooling rate,
    return the time required to reach a target temperature

    Parameters
    -----------

    Returns
    --------
    """
    time = (- 1 / k) * np.log( (T_targ - T_env) /  (T_init - T_env))
    return time


# solve the coffee question
T_cream = solve_temp(time_array, T_init=85)
T_nocrm = solve_temp(time_array, T_init=90)

# Get time to drinkable temperature
t_cream = time_to_temp(60, T_init=85)
t_nocrm = time_to_temp(60, T_init=90)
t_smart = time_to_temp(65, T_init=90)

# create figure and axes objects
fig, ax = plt.subplots(1,1)

# plot line and label
ax.plot(time_array, T_nocrm, label='No cream until cool')
ax.plot(time_array, T_cream, label='cream immediately')

ax.axvline(t_nocrm, ls='--', label='No cream: T=60')
ax.axvline(t_cream, ls='--', label='Cream: T=65')
ax.axvline(t_smart, ls='--', label='No cream: T=65')


# format axes
ax.legend(loc='best')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (degrees C)')
fig.suptitle('Time to cool a cup of coffee')

fig.tight_layout()
plt.show()