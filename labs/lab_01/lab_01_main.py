#!/usr/bin/env python3

"""
Author: Lara Tobias-Tarsh
Created: 05/09/2023

This file contains tools and scripts for repeating Lab 01
for CLaSP 410. To reproduce the plots shown in the lab report,
do this...

"""

#############
## IMPORTS ##
#############
import time
import os
import numpy as np
import matplotlib.pyplot as plt
# Let's import an object for creating our own special color map:
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

#############
## GLOBALS ##
#############

# note: 1 = burnt, 2 = tree, 3 = fire

outpath = '/Users/laratobias-tarsh/Documents/clasp410/labs/lab_01/figs'

nx, ny = 20, 20 # Number of cells in X and Y direction.
prob_spread = 1.0 # Chance to spread to adjacent cells.
prob_bare = 0.0 # Chance of cell to start as bare patch.
#prob_start = 0.1 # Chance of cell to start on fire.

# initialise forest array
forest = np.zeros([nx, ny], dtype=int) + 2
# start fire
forest[np.random.rand(nx, ny) < prob_bare] = 1

forest[nx//2 , ny//2] = 3 # force center to burn, comment out if prob_start
#forest[np.random.rand(nx, ny) < prob_start] = 3

# pad with -1 values
forest = np.pad(forest,pad_width=1,mode='constant',constant_values=-1)

idxmask = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])

frames = []
out = f'{outpath}/gridsize_{nx}_{ny}'

###############
## FUNCTIONS ##
###############

def spread_fire(array,center):
    """
    Function to spread a fire to a given quadrant of the cell

    Indexes into a given center in a 2D numpy array and assigns the surrounding
    2 x 2 grid to the values in a different array.

    Parameters
    ----------
    array : np.ndarray
        2D numpy array representing the state of the forest at given time
    center : tuple
        tuple containing index of central burning cell

    Returns
    --------
    array : np.ndarray
        2D numpy array representing the state of the forest after spreading
        fire at the given burning cell
    """
    # get new values for burning cells
    repl_array = gen_burn_mask(center[0])
    # define offsets from central cell for replacement
    offsets = np.array([[-1,-1],[-1,0],[-1,1], [0,-1],[0,0],[0,1], [1,-1],[1,0],[1,1]])
    # reshape offset array to allow easy indexing of the 2x2 grid around center
    fill = (offsets + center[:,None]).reshape(-1,2)
    # fill the centre with the replacement array values
    array[fill[:,0],fill[:,1]] = repl_array.flatten()
    return array

def gen_burn_mask(center,mask=idxmask,array=forest,p_spread=prob_spread):
    """
    Generates a mask of new cells representing a "burn pattern"
    at a given location.

    Parameters
    -----------
    mask : np.ndarray
        2D numpy array containing the spread pattern
    array : np.ndarray
        larger 2D numpy array, here a representation of a forest
    p_spread : float
        probability that the fire will spread to a given cell
    center : tuple
        tuple containing coordinates of central index of 3x3 grid around
        the burning cell
    
    Returns
    --------
    new_vals : np.ndarray
        2D array mask containing new values for fire spread
    """
    # make copy of idx array
    new_mask = np.copy(mask)
    # get grid surrounding 2 x 2 grid cell
    new_vals = array[center[0]-1:center[0]+2,center[1]-1:center[1]+2]

    # create mask dictating fire spread by comparing to random 3x3 array
    new_mask[(new_mask == 1) & (np.random.random((3,3)) < p_spread)] = 9 # arbitrary index

    new_vals[(new_mask == 9) & (new_vals == 2)] = 3 # set all values where the mask is burning to 3
    new_vals[(new_mask == 0) & (new_vals == 2)] = 2 # preserve numbering convention
    array[center[0],center[1]] = 1 # set center to 1 (burnt on this iteration)
    
    return new_vals

def plot_frames(frame,iter):
    """
    Creates a pcolor matrix plot representing the state of the 
    forest fire spread in a given forest.

    Parameters
    ----------
    frame : np.ndarray
        2D numpy array representing the state of the forest at one timestep
    iter : int
        iteration of forest fire simulation
    
    Returns
    -------
    fig : plt.Figure
        pcolor figure showing the state of the fire at a given time
    """
    # define colormap
    forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
    
    # set up figure and axes
    fig, ax = plt.subplots(1,1)
    
    # plot data at each frame
    ax.pcolor(frame, cmap=forest_cmap, vmin=1, vmax=3)
    # generate legend
    burning = mpatches.Patch(color='crimson',label='Burning')
    forested = mpatches.Patch(color='darkgreen',label='Forested')
    bare = mpatches.Patch(color='tan',label='Bare')
    plt.legend(handles=[burning,forested,bare])
    
    # format figure
    ax.set_title(f'Forest State at Iteration {iter}, ({nx} x {ny} grid)')
    fig.savefig(f'{out}/forest_iter{iter}.png')


def summary_stats(frame):
    """
    """
    # plot line plot with number of cells at each state at each iteration
    burning = (frame == 3).sum()
    forested = (frame == 2).sum()
    bare = (frame == 1).sum()
    val_dict = { "bare" : bare, "forested" : forested, "burning" : burning}
    return val_dict
    
    # plot time to spread the fire as a function of p_spread & p_bare

def timeseries_plot(ts):
    fig, ax = plt.subplots(1,1)
    ax.plot([idx for idx,dat in enumerate(ts)],
            [dat["bare"] for idx,dat in enumerate(ts)],
            label='Bare',c='tan')
    ax.plot([idx for idx,dat in enumerate(ts)],
            [dat["forested"] for idx,dat in enumerate(ts)],
            label='Forested',c='darkgreen')
    ax.plot([idx for idx,dat in enumerate(ts)],
            [dat["burning"] for idx,dat in enumerate(ts)],
            label='Burning',c='crimson')
    
    ax.set_title(f"Forest Burn Evolution with Time ({nx} x {ny} grid)")
    ax.set_xlabel("Time (iterations)")
    ax.set_ylabel("Number of Burning Cells")
    
    ax.legend()
    fig.savefig(f'{out}/burntime.png')

def main():
    """
    Executes a full simulation
    """
    iters = []
    # time code
    st = time.time()

    if not os.path.exists(out):
        os.mkdir(out)
   
    # loop over the forest until no more burning cells
    ctr = 0 # counter to keep track of while loop and pass into plots
    while 3 in forest:
        # locate index of centers where forest is burning (forest = 3)
        burning_cells = np.asarray(np.where(forest==3)).T.tolist()
        # loop over burning centers
        for center in burning_cells:
            # spread the fire at each center
            spread_fire(forest,np.array([center]))
       
        # append this to a list of frames
        frames.append(plot_frames(forest[1:-1,1:-1],ctr))   # plot frames to show forest state at each iteration
        iters.append(summary_stats(forest[1:-1,1:-1]))  # generate summary statistics of each forest fire
        
        ctr += 1 # iterate counter
    
    timeseries_plot(iters)
    et = time.time()
    elapsed_time = et - st

    print('Execution time:', elapsed_time, 'seconds')


if __name__ == "__main__":
    main()