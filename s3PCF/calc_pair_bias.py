#!/usr/bin/env python

"""
Code for computing the galaxy 3-point correlation function in the squeezed limit.
For description of the procedure can be found https://arxiv.org/abs/1705.03464.

Add to .bashrc:
export PYTHONPATH="/path/to/s3PCF:$PYTHONPATH"

"""

import numpy as np
import math
import os,sys
import random
import time
from astropy.table import Table
import astropy.io.fits as pf
from astropy.cosmology import WMAP9 as cosmo
import scipy.spatial as spatial
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
import PowerSpectrum_mod as PSm

# size of the box in box units. By definition, it is 1. 
boxsize = 1.0

# construct grid for FFTs, Ngrid controls the size of the grid
Ngrid = 512
bin_width = boxsize/Ngrid
bins = np.linspace(-boxsize/2.0, boxsize/2.0, num = Ngrid + 1)
grid_each_x = 0.5*(bins[0:-1] + bins[1:])
grid_each_y = 0.5*(bins[0:-1] + bins[1:])
grid_each_z = 0.5*(bins[0:-1] + bins[1:])
# recentered grid whose origin is at the corner of the cube
grid_each_x1 = np.concatenate((grid_each_x[Ngrid/2:], grid_each_x[:Ngrid/2]))
grid_each_y1 = np.concatenate((grid_each_y[Ngrid/2:], grid_each_y[:Ngrid/2]))
grid_each_z1 = np.concatenate((grid_each_z[Ngrid/2:], grid_each_z[:Ngrid/2]))
x_grid, y_grid, z_grid = np.meshgrid(grid_each_x1, grid_each_y1, grid_each_z1, 
    indexing='ij')

def decomp_dist(pos_full, gal_ids, params):
    """
    method that takes in pos_full, gal_ids, dist_pairs and returns 
    the los separation (z) and transverse separation (x, y)

    Parameters
    ----------
    pos_full : numpy.array
        Array of galaxy positions of shape (N, 3).

    gal_ids : numpy.array
        Array of galaxy indices of shape (N, 2). The two columns correspond 
        to the two galaxies in a pair. 
        
    params : dict
        Dictionary of simulation parameters. 

    Returns
    -------
    diff_pos_los : numpy.array
        Array of pair separations along the line of sight.

    diff_pos_trans : numpy.array
        Array of pair separations perpendicular to the line of sight.

    """

    ids1 = gal_ids[:,0].astype(int)
    ids2 = gal_ids[:,1].astype(int)
    gal_pos1 = pos_full[ids1]
    gal_pos2 = pos_full[ids2]
    diff_pos = gal_pos1 - gal_pos2 

    # calculate los distance and transverse distance
    diff_pos_los = np.absolute(diff_pos[:,2])
    diff_pos_trans = np.sqrt(diff_pos[:,0]**2+diff_pos[:,1]**2)

    return diff_pos_los, diff_pos_trans

def cast_TSC(pos_array, gridsize = Ngrid):
    """
    method that casts an array of 3D positions onto a TSC grid.

    Parameters
    ----------
    pos_array : numpy.array
        Array of galaxy positions of shape (N, 3).

    gridsize : int
        Size of the grid on each side, default set to 512. 

    Returns
    -------
    PSm.CalculateTSC(pos_array, gridsize, boxsize): numpy.array
        The TSC grid. 

    """

    return PSm.CalculateTSC(pos_array, gridsize, boxsize)

def gauss(x, y, z, sigx, sigy, sigz):
    """
    method that returns a custom gaussian centered on the origin.

    Parameters
    ----------
    x, y, z : float
        Input coordinates.

    sigx, sigy, sigz : float
        Width of the Gaussian in all 3 directions. 

    Returns
    -------
    the value of the Gaussian

    """
    return np.exp(-0.5*(x**2/sigx**2+y**2/sigy**2+z**2/sigz**2))

def sigmoid(x, y, z, r0, s):
    """
    method that returns a custom sigmoid.

    Parameters
    ----------
    x, y, z : float
        Input coordinates.

    r0 : float
        The central location of the sigmoid.

    s : float
        The slope of the sigmoid. 

    Returns
    -------
    the value of the sigmoid

    """
    return 1.0/(1+np.exp(-s*(np.sqrt(x**2 + y**2) - r0)))

def window_gauss_sigmoid(x, y, z, params, rsd):
    """
    method that builds a gaussian-sigmoid window function.

    Parameters
    ----------
    x, y, z : float
        Input coordinates.

    params : dict
        Dictionary of simulation parameters. 

    rsd : boolean
        Flag 'True' if we are invoking RSD. 

    Returns
    -------
    The value of the window function

    """

    # if we invoke RSD, then we construct a larger window function 
    if rsd:
        gauss_sig = 50 # cell length, Mpc
        r0cell = 35 # cell length, Mpc 
        s0 = 0.5 # Mpc^ -1
        cutoff = 20 # Mpc
    else:
        gauss_sig = 30 # cell length, Mpc
        r0cell = 20 # cell length, Mpc 
        s0 = 0.8 # Mpc^ -1
        cutoff = 10 # Mpc 

    sigx = gauss_sig / params['Lbox'] # 20 Mpc
    sigy = gauss_sig / params['Lbox']
    sigz = gauss_sig / params['Lbox']
    part_gauss = gauss(x, y, z, sigx, sigy, sigz)
    r0 = r0cell/params['Lbox']
    part_sigmoid = sigmoid(x, y, z, r0, s0*params['Lbox'])
    # we are also introducing a cutoff that cleans up the cavity 
    # in the middle. 
    part_turnoff = np.sqrt(x**2 + y**2) > cutoff / params['Lbox']
    return part_gauss*part_sigmoid*part_turnoff

def gal_convolve_W(gal_grid, W_fft, params, saveout = False):
    """
    method that convolves the galaxy density field with the window function
    using explicit FFTs.

    Parameters
    ----------
    gal_grid : numpy.array
        The galaxy density field. 

    W_fft : numpy.array
        The FFT of the window function. 

    params : dict
        Dictionary of simulation parameters. 

    saveout : boolean
        Flag 'True' if we are saving the output. 

    Returns
    -------
    gal_grid_convolved : numpy.array
        The convolved galaxy density field.

    gal_fft : numpy.array
        The FFT of the galaxy density field. 

    """

    # do convolution by explict ffts
    print "Starting Convolution (manual)..."
    start = time.time()

    gal_fft = np.fft.rfftn(gal_grid)

    gal_grid_convolved = np.fft.irfftn(gal_fft*W_fft, gal_grid.shape)
    print "Convolution done, time elapsed: ", time.time() - start

    # save the convolved fields
    if saveout:
        datadir = './data'
        if params['rsd']:
            datadir = datadir+'_rsd'
        np.save(datadir+'/gal_grid_convolved_manual', gal_grid_convolved)
        np.save(datadir+'/gal_grid_fft', gal_fft)
    return gal_grid_convolved, gal_fft

def pair_convolve_W(pair_grid, W_fft, params, saveout = False):
    """
    method that convolves the galaxy density field with the window function
    using explicit FFTs.

    Parameters
    ----------
    gal_grid : numpy.array
        The galaxy density field. 

    W_fft : numpy.array
        The FFT of the window function. 

    params : dict
        Dictionary of simulation parameters. 

    saveout : boolean
        Flag 'True' if we are saving the output. 

    Returns
    -------
    gal_grid_convolved : numpy.array
        The convolved galaxy density field.

    gal_fft : numpy.array
        The FFT of the galaxy density field. 

    """

    # do convolution by explict ffts
    print "Starting Convolution (manual)..."
    start = time.time()

    pair_fft = np.fft.rfftn(pair_grid)

    pair_grid_convolved = np.fft.irfftn(pair_fft*W_fft, pair_grid.shape)
    print "Convolution done, time elapsed: ", time.time() - start

    # save the convolved fields
    if saveout:
        datadir = './data'
        if params['rsd']:
            datadir = datadir+'_rsd'
        np.save(datadir+'/pair_grid_convolved_manual', pair_grid_convolved)
        np.save(datadir+'/pair_grid_fft', pair_fft)
    return pair_grid_convolved, pair_fft

def interp_odens(pos, odens_field, params):
    """
    method that function that takes in a position and a overdensity field
    and then returns an interpolated overdensity at that position.

    Parameters
    ----------
    pos : numpy.array
        The 3D position to be interpolated.  

    odens_field : numpy.array
        The overdensity field. 

    params : dict
        Dictionary of simulation parameters. 

    Returns
    -------
    The interpolated value at the position specified by pos. 

    """
    x, y, z = pos

    # find the nearerest elements
    ix = int((x + 0.5 - bin_width/2)/bin_width)
    if ix == -1:
        ix = Ngrid - 1
    iy = int((y + 0.5 - bin_width/2)/bin_width)
    if iy == -1:
        iy = Ngrid - 1
    iz = int((z + 0.5 - bin_width/2)/bin_width)
    if iz == -1:
        iz = Ngrid - 1
    ix1 = (ix + 1) % Ngrid
    iy1 = (iy + 1) % Ngrid
    iz1 = (iz + 1) % Ngrid

    # pull out the nearest elements
    tot_dens = 0
    tot_dist = 0
    for ex in [ix, ix1]:
        for ey in [iy, iy1]:
            for ez in [iz, iz1]:
                del_x = x - grid_each_x[ex]
                del_y = y - grid_each_y[ey]
                del_z = z - grid_each_z[ez]
                # update the values if they are wrapping around
                if ix == Ngrid - 1:
                    del_x = x - (grid_each_x[ix] + bin_width)
                if iy == Ngrid - 1:
                    del_y = y - (grid_each_y[iy] + bin_width)
                if iz == Ngrid - 1:
                    del_z = z - (grid_each_z[iz] + bin_width)
                # update total dens and total dist
                dist = np.sqrt(del_x**2 + del_y**2 + del_z**2)
                tot_dens += dist*odens_field[ex, ey, ez]
                tot_dist += dist

    return tot_dens/tot_dist

def calc_qeff(whichsim, pos_full, pair_data, params, 
    dist_nbins = 30, whatseed = 0, rsd = True):
    """
    main method that computes the galaxy pair bias and the squeezed 3PCF.

    Parameters
    ----------
    whichsim : int
        Which of the simulation boxes we are computing for. 

    pos_full : numpy.array
        Array of galaxy positions of shape (N, 3).

    pair_data : numpy.array
        Array of pair data containing the following seven columns:
        x (Mpc), y (Mpc), z (Mpc), dist (Mpc), id1, id2, mhalo (Msun)
        The ids are the galaxy id number of the two galaxies in the pair. 

    params : dict
        Dictionary of simulation parameters. 

    dist_nbins : int
        Number of bins in pair separation along the parallel and 
        perpendicular direction. 

    whatseed : int
        The seed to the random number generator. 

    rsd : boolean
        Flag 'True' if we are invoking RSD. 

    Outputs
    -------
    Save a text file that contains the following columns:
    ilos, itrans, bpg, Qeff, npairs

    ilos   : the bin index along the line of sight.
    itrans : the bin index perpendicular to the line of sight. 
    bpg    : the pair-galaxy bias for the bin. 
    Qeff   : the squeezed 3PCF for the bin.
    npairs : the number of pairs in the bin. 

    """

    # pos_ful has to be three columns in Mpc, from 0 to L_box
    print "Calculating bias. Realization: ", whichsim

    # convert to box units from -0.5 to 0.5
    pos_pairs = pair_data[:,0:3] / params['Lbox'] - 0.5
    dist_pairs = pair_data[:,3] / params['Lbox']
    gal_ids = pair_data[:,4:]
    # decompose the pair separation into a los component and a trans component
    dist_los, dist_trans = decomp_dist(pos_full, gal_ids, params)

    # cast the gal distribution onto a 3D grid using TSC
    gal_grid = cast_TSC(pos_full, Ngrid)
    Ngals = len(pos_full)

    # calculate window function and its fft
    print "Setting up window function and its fft..."
    start = time.time()
    # check for window function directory
    wdir = "./windows"
    if rsd:
        wdir = wdir+"_rsd"
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    wfft_filename = wdir+"/wfft"
    print os.path.isfile(wfft_filename+".npy")
    # if file doesnt exist, calculate the fft, but if it does, just load file
    if os.path.isfile(wfft_filename+".npy"):
        W_fft = np.load(wfft_filename+".npy")
    else:
        print "building window function and its FFT"
        W_grid = window_gauss_sigmoid(x_grid, y_grid, z_grid, params, rsd)
        # normalize window function
        W_grid_norm = W_grid/np.sum(W_grid)
        # fft the window function
        W_fft = np.fft.rfftn(W_grid_norm)
        np.save(wfft_filename, W_fft)

    print "Time elapsed: ", time.time() - start
    
    # convolve the post tsc grid with the window function
    gal_grid_convolved, gal_fft = gal_convolve_W(gal_grid, W_fft, params)

    print "Calculating galaxy bias..."
    # calculating overdensity fields
    rho_gal_avg = np.mean(gal_grid_convolved)
    delta_gal = (gal_grid_convolved - rho_gal_avg)/rho_gal_avg

    # weighted average of the galaxy overdensity field
    b_gals = np.sum(delta_gal*gal_grid)/np.sum(gal_grid)
    print "b_gals = ", b_gals

    # create bins for dist_los and dist_trans
    bins_los = np.linspace(0,params['maxdist']/params['Lbox'],
        num=dist_nbins + 1)
    bins_trans = np.linspace(0,params['maxdist']/params['Lbox']/3,
        num=dist_nbins + 1)
    bins_trans_lo = bins_trans[:-1]
    bins_trans_hi = bins_trans[1:]

    # open a file to store outputs
    filename = "./data_"+str(dist_nbins)+"_"+str(dist_nbins)
    +"_ilos_itrans_maxdist"+str(int(params['maxdist']))+"_sim"+str(whichsim)
    +".txt"
    f = open(filename,'w')
    f.write("%s \n" % (b_gals))
    f.close()

    # for each ilos, loop through itrans to fill in bs_pair
    for ilos in range(0, dist_nbins):
        for itrans in range(0, dist_nbins):
            # find the indices of all the pairs that fall within this dist bin
            masks_los = [dist_los > bins_los[ilos], dist_los < bins_los[ilos+1]]
            masks_trans = [dist_trans > bins_trans[itrans], 
                           dist_trans < bins_trans[itrans+1]]
            allmasks = masks_los + masks_trans
            tot_mask = reduce(np.logical_and, allmasks)
            tot_mask = np.array(tot_mask)
            npairs = np.sum(tot_mask) # number of pairs in this bin
            # if the sub sample is empty, skip
            if np.sum(tot_mask) == 0:
                continue
            # pull out the subsample of pairs
            pos_pairs_sub = pos_pairs[tot_mask]
            dist_pairs_sub = dist_pairs[tot_mask]
            gal_ids_sub = gal_ids[tot_mask]

            # calculating average overdensity of pairs
            # calculate the list of odens at all pairs
            odens_all_pairs = np.zeros(len(pos_pairs_sub))
            for i in range(0, len(pos_pairs_sub)):
                odens_all_pairs[i] = interp_odens(pos_pairs_sub[i], 
                                                    delta_gal, params)
            # average b_pair averaged over all pairs
            b_pair = np.mean(odens_all_pairs)

            # compute the 2PCF so that we can compute the Qeff
            ndens = Ngals/params['Lbox']**3 # per mpc^3
            del_d_los = abs(np.mean(np.diff(bins_los)))
            del_A = np.pi*(bins_trans_hi[itrans]**2 - bins_trans_lo[itrans]**2)
            n_expected = del_A*2*del_d_los*ndens
            gal_corr = npairs/n_expected/(Ngals/2.0) - 1.0

            # compute the bpg
            bpg = b_pair/b_gals

            # correction term that depends on maxdist
            if params['maxdist'] == 10:
                galcorr_dw = 0.362
            elif params['maxdist'] == 30:
                galcorr_dw = 0.104
            else:
                galcorr_dw = 0
                print "galcorr_dw is unknown for this maxdist."

            # compute the Qeff
            Qeff = ((1+gal_corr)*bpg - 2 - galcorr_dw/(2*gal_corr))/\
            (2*gal_corr + 1.5*galcorr_dw)

            # output data
            f = open(filename, 'a')
            f.write("%s %s %s %s %s \n" % (ilos, 
                                           itrans, 
                                           bpg,
                                           Qeff, 
                                           np.sum(tot_mask)))
            f.close()
            




