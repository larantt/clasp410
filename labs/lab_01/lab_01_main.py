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
import random
# Let's import an object for creating our own special color map:
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib as mpl

#############
## GLOBALS ##
#############

# note: 1 = burnt, 2 = tree, 3 = fire

# USER INPUT
outpath = '/Users/laratobias-tarsh/Documents/clasp410/labs/lab_01/figs'

nx, ny = 20, 40 # Number of cells in X and Y direction.


# set probabilities with range 0-1 for questions 2 and 3, step size 0.05 so that they are iterable
prob_spread = 0.5 # Chance to spread to adjacent cells.
prob_bare = 0 # Chance of cell to start as bare patch.
prob_start = 0.1 # Chance of cell to start on f
prob_fatal = 0.2
question = 'question_2' # file management for IO

probs = np.round(np.linspace(0, 1, 11),2) # Chance to spread to adjacent cells.

start_rand = True      # change for when start_rand should be false
output_frames = False   # chage if you want all the output frames

n_iters = 1
Mode = "forest" #"forest"

# DEFAULTS
# define mask for indexing neighbour cells
idxmask = np.array([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]])

out = f'{outpath}/{question}/gridsize_{nx}_{ny}'

###############
## FUNCTIONS ##
###############

########## ALGORITHM FUNCTIONS ##############
def fire_init(p_bare=prob_bare,p_start=prob_start):
    """
    """
    # initialise forest array
    forest = np.zeros([nx, ny], dtype=int) + 2
    # start fire
    forest[np.random.rand(nx, ny) < p_bare] = 1

    if start_rand is False:
        forest[nx//2 , ny//2] = 3 # force center to burn, comment out if prob_start
    else :
        forest[np.random.rand(nx, ny) < p_start] = 3

    # pad with -1 values
    forest = np.pad(forest,pad_width=1,mode='constant',constant_values=-1)
    return forest

def spread_fire(array,center,p_spread):
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
    repl_array = gen_burn_mask(center[0],array,p_spread)
    # define offsets from central cell for replacement
    offsets = np.array([[-1,-1],[-1,0],[-1,1], [0,-1],[0,0],[0,1], [1,-1],[1,0],[1,1]])
    # reshape offset array to allow easy indexing of the 2x2 grid around center
    fill = (offsets + center[:,None]).reshape(-1,2)
    # fill the centre with the replacement array values
    array[fill[:,0],fill[:,1]] = repl_array.flatten()
    return array

def gen_burn_mask(center,array,p_spread,mask=idxmask,):
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
    
    if Mode == 'disease':
        status = 0 if random.uniform(0,1) < prob_fatal else 1
        array[center[0],center[1]] = status # set center to 1 (burnt on this iteration)
    else:
        array[center[0],center[1]] = 1 # set center to 1 (burnt on this iteration)
    return new_vals

def full_sim(outpath,forest,p_spread=prob_spread):
    """
    Executes a full simulation
    """
    # initialise the forest
    #forest = fire_init(p_bare,p_start,start_rand)
    
    # empty lists for storage
    iters = []          # values at each iteration
    frames = []         # plot of forest at each state
   
    # loop over the forest until no more burning cells
    ctr = 0 # counter to keep track of while loop and pass into plots

    # append the first frame
    frames.append(plot_frames(forest[1:-1,1:-1],ctr,outpath))   # plot frames to show forest state at each iteration
    iters.append(summary_stats(forest[1:-1,1:-1]))  # generate summary statistics of each forest fire iteration

    # get initial amount of forest
    init_forested = (forest == 2).sum()
    
    while 3 in forest:
        ctr += 1 # iterate counter
        # locate index of centers where forest is burning (forest = 3)
        burning_cells = np.asarray(np.where(forest==3)).T.tolist()
        # loop over burning centers
        for center in burning_cells:
            # spread the fire at each center
            spread_fire(forest,np.array([center]),p_spread)
       
        # append this to a list of frames
        if output_frames is True:
            frames.append(plot_frames(forest[1:-1,1:-1],ctr,outpath))   # plot frames to show forest state at each iteration
        iters.append(summary_stats(forest[1:-1,1:-1]))  # generate summary statistics of each forest fire iteration
        
    if Mode == 'forest':
        timeseries_plot(iters,outpath)

        # get percentage of forest burned
        end_forested = (forest == 2).sum()
        percent_burned = ((end_forested - init_forested) / init_forested) * 100
    else:
        mortality_rate = ((forest == 0).sum())/(nx * ny)
        percent_burned = mortality_rate * 100
    
    # return number of iterations to complete the simulation based on the initial parameters
    return ctr,iters,percent_burned

############# ANALYSIS FUNCTIONS ######################
def plot_frames(frame,iter,outpath):
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
    forest_cmap = ListedColormap(['black','tan', 'darkgreen', 'crimson'])
    
    # set up figure and axes
    fig, ax = plt.subplots(1,1)
    if Mode == 'disease':
    # plot data at each frame
        ax.pcolor(frame, cmap=forest_cmap, vmin=0, vmax=3)

            # generate legend
        infected = mpatches.Patch(color='crimson',label='infected')
        healthy = mpatches.Patch(color='darkgreen',label='healthy')
        immune = mpatches.Patch(color='tan',label='immune')
        dead = mpatches.Patch(color='black',label='Dead')
        ax.legend(handles=[infected,healthy,immune,dead])
        
        # format figure
        ax.set_title(f'Disease State at Iteration {iter}, ({nx} x {ny} grid)')
        fig.savefig(f'{outpath}/forest_iter{iter}.png')
        plt.close()
    else:
        ax.pcolor(frame, cmap=forest_cmap, vmin=1, vmax=3)
    # generate legend
        burning = mpatches.Patch(color='crimson',label='Burning')
        forested = mpatches.Patch(color='darkgreen',label='Forested')
        bare = mpatches.Patch(color='tan',label='Bare')
        ax.legend(handles=[burning,forested,bare])
        
        # format figure
        ax.set_title(f'Forest State at Iteration {iter}, ({nx} x {ny} grid)')
        fig.savefig(f'{outpath}/forest_iter{iter}.png')
        plt.close()

def vary_params(param):
    """
    """
    # make list to store ensembles
    prob_ens = []
    prob_finish = []
    prob_burned = []

    # iterate over prob with fixed p_bare = 0 -> scatter plot of time to complete w regression
    forest = fire_init()
    for prob in probs:

        # I LOVE FILE MANAGEMENT YAY I LOVE FILE MANAGEMENT YAY
        outpath_iter = f'{out}/prob__{prob}'
        if not os.path.exists(outpath_iter):
            os.mkdir(outpath_iter)
        # run simulation at given prob value
        if param == 'p_spread':
            forest2 = forest.copy()
            time_to_finish, run_values, percent_burned = full_sim(outpath_iter,forest2,p_spread=prob)
        
        elif param == 'p_bare':
            forest = fire_init(p_bare=prob)
            time_to_finish, run_values, percent_burned = full_sim(outpath_iter,forest)
        
        elif param == 'p_start':
            forest = fire_init(p_start=prob)
            time_to_finish, run_values, percent_burned = full_sim(outpath_iter,forest)
        else:
            raise Exception("choose valid probability from: p_spread, p_bare, p_start")
        # store the values in a list
        prob_finish.append(time_to_finish)
        prob_ens.append(run_values)
        prob_burned.append(percent_burned)

    return prob_ens,prob_burned,prob_finish

def run_ensemble(n_iters,param='p_spread'):
    """
    """

    fig1,ax1 = plt.subplots(1,1)
    
    ens_ts = []
    burned_ts = []
    finish_ts = []
    
    for i in range(n_iters):
        ens,burned,finish = vary_params(param)
        ax1.scatter(probs,finish,alpha=0.5)
        ens_ts.append(ens)
        burned_ts.append(burned)
        finish_ts.append(finish)
    
    # format figure
    ax1.set_title(f'Relationship between {param} and simulation completion\n ({nx} x {ny} Grid)')
    ax1.set_title(f'n iterations = {n_iters}',loc='right',c='k')
    ax1.set_xlabel(param)
    ax1.set_ylabel('Time to Complete Simulation (iterations)')
    fig1.tight_layout()
    fig1.savefig(f'{out}/{param}_scatter.png')

    return ens_ts, burned_ts, finish_ts

############## PLOTTING FUNCTIONS ######################
def summary_stats(frame):
    """
    """
    # plot line plot with number of cells at each state at each iteration
    burning = (frame == 3).sum()
    forested = (frame == 2).sum()
    bare = (frame == 1).sum()
    dead = (frame == 0).sum()
    if Mode == 'disease':
        val_dict = { "immune" : bare, "healthy" : forested, "infected" : burning, "dead" : dead}
    else:
        val_dict = { "bare" : bare, "forested" : forested, "burning" : burning}
    return val_dict
    
    # plot time to spread the fire as a function of p_spread & p_bare

def timeseries_plot(ts,outpath):
    """"""
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
    
    fig.suptitle("Forest Burn Evolution with Time")
    ax.set_title(f"{nx} x {ny} grid",loc='left')
    ax.set_title(f"")
    ax.set_xlabel("Time (iterations)")
    ax.set_ylabel("Number of Burning Cells")
    
    ax.legend()
    fig.savefig(f'{outpath}/burntime.png')

def ens_timeseries_plot(ts_list,num_sim):
    """"""
    # iterate over number of trials
    for ts in ts_list:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10), sharex=True)
        colors = ['#46f0f0', '#f032e6', '#fabebe', '#008080', '#e6beff', 
                  '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',]
        # iterate over p_spread
        for idx,list in enumerate(ts):
                # plot bare data
            if Mode == 'forest':
                ax1.plot([idx for idx,dat in enumerate(list)],
                        [((dat["bare"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label='Bare',c='tan',alpha=0.5)
                
                ax2.plot([idx for idx,dat in enumerate(list)],
                        [((dat["bare"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label=probs[idx],
                        c=colors[idx],alpha=0.5)
                
                # plot forested data
                ax1.plot([idx for idx,dat in enumerate(list)],
                        [((dat["forested"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label='Forested',c='darkgreen',alpha=0.5)
                
                ax3.plot([idx for idx,dat in enumerate(list)],
                        [((dat["forested"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label=probs[idx],
                        c=colors[idx],alpha=0.5)
                
                # plot burning data
                ax1.plot([idx for idx,dat in enumerate(list)],
                        [((dat["burning"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label='Burning',c='crimson',alpha=0.5)
                
                ax4.plot([idx for idx,dat in enumerate(list)],
                        [((dat["burning"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label=probs[idx],
                        c=colors[idx],alpha=0.5)
            
            elif Mode == 'disease':
                
                ax1.plot([idx for idx,dat in enumerate(list)],
                        [((dat["dead"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label=probs[idx],
                        c=colors[idx],alpha=0.5)


                ax2.plot([idx for idx,dat in enumerate(list)],
                        [((dat["immune"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label=probs[idx],
                        c=colors[idx],alpha=0.5)
                
                ax3.plot([idx for idx,dat in enumerate(list)],
                        [((dat["healthy"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label=probs[idx],
                        c=colors[idx],alpha=0.5)
                
                ax4.plot([idx for idx,dat in enumerate(list)],
                        [((dat["infected"] / (nx * ny)) * 100) for idx,dat in enumerate(list)],
                        label=probs[idx],
                        c=colors[idx],alpha=0.5)
                

    if Mode == 'disease':            
        ax1.set_ylabel("Percentage of Population Dead (%)")
        ax2.set_ylabel("Percentage of Population Immune (%)")
        ax3.set_ylabel("Percentage of Population Healthy (%)")
        ax4.set_ylabel("Percentage of Population Infected(%)")

        ax1.set_title("Percentage of Population Dead")
        ax2.set_title("Percentage of Population Immune")
        ax3.set_title("Percentage of Population Healthy")
        ax4.set_title("Percentage of Population Infected")

        ax1.legend([probs[i] for i,val in enumerate(colors)],title='p_start',loc='upper right')
    else:
        ax1.set_ylabel("Percentage of Cells (%)")
        ax2.set_ylabel("Percentage of Bare Cells (%)")
        ax3.set_ylabel("Percentage of Forested Cells (%)")
        ax4.set_ylabel("Percentage of Burning Cells (%)")

        ax1.set_title("All Variables")
        ax2.set_title("Bare Cells")
        ax3.set_title("Forested Cells")
        ax4.set_title("Burning Cells")

        # generate legend
        burning = mpatches.Patch(color='crimson',label='Burning')
        forested = mpatches.Patch(color='darkgreen',label='Forested')
        bare = mpatches.Patch(color='tan',label='Bare')
        ax1.legend(handles=[burning,forested,bare])

    ax2.legend([probs[i] for i,val in enumerate(colors)],title='p_spread',loc='upper right')
    ax3.legend([probs[i] for i,val in enumerate(colors)],title='p_spread',loc='upper right')
    ax4.legend([probs[i] for i,val in enumerate(colors)],title='p_spread',loc='upper right')
                

    fig.suptitle(f"{Mode.capitalize()} Evolution with Time ({nx} x {ny} grid) ")

    #ax1.set_xlabel("Time (iterations)")

    fig.text(0.55, 0.001, 'Time (iterations)', ha='center')

    fig.tight_layout()
    

    fig.savefig(f'{out}/burntime_ens_{num_sim}.png')

def violin_plots(timeseries,param):
    """
    """
    data = []
    for idx in range(len(probs)):
        data.append(list(timeseries[:,idx]))
    
    fig,ax = plt.subplots(1,1)
    ax.violinplot(data[::-1],positions=[11,10,9,8,7,6,5,4,3,2,1],showmeans=True)

    ax.set_xticks(np.arange(1, len(probs) + 1))
    ax.set_xticklabels(probs)
    ax.set_xlim(0.25, len(probs) + 0.75)

    ax.set_title(f'({nx} , {ny}) Grid', loc='left')
    ax.set_title(f'{len(timeseries)} iterations', loc='right')
    ax.set_title(f'p_spread = {prob_spread}',loc='center')

    if Mode == 'forest':
        ax.set_xlabel('Probability of Bare Cells at Start of Simulation')
        ax.set_ylabel('Change in Forest (%)')

        fig.suptitle('Relationship Between p_bare and Percentage Change in Forest')
    if Mode == 'disease':
        ax.set_xlabel('Probability of Vaccination Per Person')
        ax.set_ylabel('Percentage of Population Dead (%)')

        fig.suptitle('Relationship Between Vaccine Uptake and Mortality')


    fig.savefig(f'{out}/{param}_violin.png')

############### EXECUTE PROGRAM ########################
def main():
    """
    Main function for forest fire simulation.

    Allows for the generation of summary plots based on
    the variation of initial parameters.
    """
    # make directory if it doesnt exist to store figures
    if not os.path.exists(out):
        os.mkdir(out)
    
    # QUESTION 1:
    # run 3x3 forest simulation with start_rand = False
    # run 6 x 4 forest simulation with start_rand = False, change nx=6 and ny=4
    if question == 'question_1':
        q1_forest = fire_init()
        full_sim(out,q1_forest)

    # QUESTION 2:
    if question == 'question_2':
        # vary p_spread to get dependence of spread rate
        spread_en_ts, spread_burned_ts, spread_finish_ts = run_ensemble(1,'p_spread')
        # run ensemble of varied p_bare to get distribution of remaining forest dependent on bare spots
        bare_en_ts, bare_burned_ts, bare_finish_ts = run_ensemble(50,'p_bare')
        # plot line plot showing dependence on spread rate
        ens_timeseries_plot(spread_en_ts,'p_spread')
        # plot violin distributions showing dependence on vaccine uptake
        violin_plots(np.array(bare_burned_ts),prob_bare)


    # QUESTION 3:
    if question == 'question_3':
        # vary p_start to get dependence of mortality rate
        start_en_ts, start_burned_ts, start_finish_ts = run_ensemble(1,'p_start')
        # vary p_spread to get dependence of spread rate
        spread_en_ts, spread_burned_ts, spread_finish_ts = run_ensemble(1,'p_spread')
        # run ensemble of varied p_bare to get distribution of death rate dependent of vaccination
        bare_en_ts, bare_burned_ts, bare_finish_ts = run_ensemble(50,'p_bare')
        # plot line plot showing dependence on spread rate
        ens_timeseries_plot(spread_en_ts,'p_spread')
        # plot line plot showing dependence on mortality rate
        ens_timeseries_plot(start_en_ts,'p_start')
        # plot violin distributions showing dependence on vaccine uptake
        violin_plots(np.array(bare_burned_ts),prob_bare)
    
    

if __name__ == "__main__":
    main()