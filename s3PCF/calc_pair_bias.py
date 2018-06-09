#allsims testhod
import numpy as np
import math
import os,sys
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import rc, rcParams
rcParams.update({'font.size': 20})
from mpl_toolkits.mplot3d import Axes3D
from astropy.table import Table
import astropy.io.fits as pf
from astropy.cosmology import WMAP9 as cosmo
import multiprocessing
from multiprocessing import Pool
import scipy.spatial as spatial
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
import PowerSpectrum_mod as PSm

boxsize = 1.0

# define grid
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
x_grid, y_grid, z_grid = np.meshgrid(grid_each_x1, grid_each_y1, grid_each_z1, indexing='ij')

# method that takes in pos_full, gal_ids, dist_pairs and returns 
# the los separation (z) and transverse separation (x, y)
def decomp_dist(pos_full, gal_ids, dist_pairs, params):
    ids1 = gal_ids[:,0].astype(int)
    ids2 = gal_ids[:,1].astype(int)
    gal_pos1 = pos_full[ids1]
    gal_pos2 = pos_full[ids2]
    diff_pos = gal_pos1 - gal_pos2 # x, y, z
    # calculate los distance and transverse distance
    diff_pos_los = np.absolute(diff_pos[:,2])
    diff_pos_trans = np.sqrt(diff_pos[:,0]**2+diff_pos[:,1]**2)
    return diff_pos_los, diff_pos_trans


######## cast the galaxy pair distribution onto a grid using TSC
# method that creats the TSC density field
def cast_TSC(pos_array, gridsize = Ngrid):
    # call PSm which is done in C++, within Abacus
    return PSm.CalculateTSC(pos_array, gridsize, boxsize)

######## convolve the gridded galaxy pair distribution with a window function
# for gaussian-sigmoid

# rsd
gauss_sig = 50 # cell length, Mpc
r0cell = 35 # cell length, Mpc 
s0 = 0.5 # Mpc^ -1
# for sigmoid-exp
cutoff = 20 # Mpc

"""
# no rsd
gauss_sig = 30 # cell length, Mpc
r0cell = 20 # cell length, Mpc 
s0 = 0.8 # Mpc^ -1
# for sigmoid-exp
cutoff = 10 # Mpc
"""
wtype = "_gauss_sigmoid"
wparams = "_"+str(int(gauss_sig))+"_"+str(int(r0cell))+"_"+str(int(s0*10))+"_"+str(int(cutoff))

# define window function, ready to convolve with galaxy grid
def gauss(x, y, z, sigx, sigy, sigz):
    return np.exp(-0.5*(x**2/sigx**2+y**2/sigy**2+z**2/sigz**2))

# sigmoid function
def sigmoid(x, y, z, r0, s):
    # s is slope
    # r0 is the radius of the hole in the middle
    return 1.0/(1+np.exp(-s*(np.sqrt(x**2 + y**2) - r0)))

def window_gauss_sigmoid(x, y, z, params):
    sigx = gauss_sig / params['Lbox'] # 20 Mpc
    sigy = gauss_sig / params['Lbox']
    sigz = gauss_sig / params['Lbox']
    part_gauss = gauss(x, y, z, sigx, sigy, sigz)
    r0 = r0cell/params['Lbox']
    part_sigmoid = sigmoid(x, y, z, r0, s0*params['Lbox'])
    part_turnoff = np.sqrt(x**2 + y**2) > cutoff / params['Lbox']
    return part_gauss*part_sigmoid*part_turnoff

# function that convolves gal density field with W function using explicit fft
def gal_convolve_W(gal_grid, W_fft, params, saveout = False, makeplots = False):

    # do convolution by explict ffts
    print "Starting Convolution (manual)..."
    start = time.time()

    gal_fft = np.fft.rfftn(gal_grid)

    gal_grid_convolved = np.fft.irfftn(gal_fft*W_fft, gal_grid.shape)
    print "Convolution done, time elapsed: ", time.time() - start
    print "Check if galaxy count is conserved:"
    print "Galaxy count before convolution:", np.sum(gal_grid)
    print "Galaxy count after convolution:", np.sum(gal_grid_convolved)

    # save the convolved fields
    if saveout:
        datadir = './data'
        if params['rsd']:
            datadir = datadir+'_rsd'
        np.save(datadir+'/gal_grid_convolved_manual', gal_grid_convolved)
        np.save(datadir+'/gal_grid_fft', gal_fft)
    # make plots
    if makeplots:
        fig = pl.figure(1)
        pl.clf()
        pl.imshow(gal_grid_convolved[0])
        pl.colorbar()
        plotdir = "./plots"
        if params['rsd']:
            plotdir = plotdir + "_rsd"
        fig.savefig(plotdir+"/plot_gal_convolved_manual_slice.png")
    return gal_grid_convolved, gal_fft

# function that convolves pair density field with W function using explicit fft
def pair_convolve_W(pair_grid, W_fft, params, saveout = False, makeplots = False):

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
    # make plots
    if makeplots:
        fig = pl.figure(2)
        pl.clf()
        pl.imshow(pair_grid_convolved[0])
        plotdir = "./plots"
        if params['rsd']:
            plotdir = plotdir + "_rsd"
        fig.savefig(plotdir+"/plot_pair_convolved_manual_slice.png")
    return pair_grid_convolved, pair_fft

######### calculate pair/galaxy bias ratio
# calculate galaxy overdensity at each galaxy/pair location
# For galaxy: we average over all grid points, weighted by the unconvolved field.
# For pairs: we average over pairs, with overdensities interpolated between the nearest cells.

# function that takes in a position and a overdensity field
# and then returns a interpolated overdensity at that position
def interp_odens(pos, odens_field, params):
    
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


# this is the main method, call this to run this whole program
def calc_bias(whichsim, pos_ful, pair_data, params, dist_nbins = 30, whatseed = 0, rsd = True):
    # pos_ful has to be three columns in Mpc, from 0 to L_box
    # pair_data has to be seven columns
    # x (Mpc), y (Mpc), z (Mpc), dist (Mpc), id1, id2, mhalo (Msun)
    print "Calculating bias. Realization: ", whichsim

    M_cut, M1, sigma, alpha, kappa, A = design
    
    datadir = "./data"
    if rsd:
        datadir = datadir+"_rsd"
    #savedir = datadir+"/rockstar_"+str(M_cut)[0:4]+"_"+str(M1)[0:4]+"_"+str(sigma)[0:4]+"_"+str(alpha)[0:4]+"_"+str(kappa)[0:4]+"_"+str(A)
    savedir = datadir+"/newdata"
    if rsd:
        savedir = savedir+"_rsd"
    # if we are doing repeats, save them in separate directories
    if not whatseed == 0:
        savedir = savedir+"_"+str(whatseed)

    """
    # load the galaxy and pair catalog
    print "Loading pair/galaxy catalogs...", whichsim
    pos_full = np.fromfile(savedir+"/halos_gal_full_pos_"+str(whichsim))
    pair_data = np.fromfile(savedir+"/halos_pairs_full_"+str(whichsim)+"_maxdist"+str(int(params['maxdist'])))
    pos_full = np.array(np.reshape(pos_full, (-1, 3))) / params['Lbox'] - 0.5 # relative unit
    pair_data = np.array(np.reshape(pair_data, (-1, 7)))
    """

    # convert to box units from -0.5 to 0.5
    pos_pairs = pair_data[:,0:3] / params['Lbox'] - 0.5
    dist_pairs = pair_data[:,3] / params['Lbox']
    gal_ids = pair_data[:,4:]
    # decompose the pair separation into a los component and a trans component
    dist_los, dist_trans = decomp_dist(pos_full, gal_ids, dist_pairs, params)

    # calculate galaxy bias 
    # cast the gal distribution onto a 3D grid using TSC
    gal_grid = cast_TSC(pos_full, Ngrid)
    Ngals = np.sum(gal_grid)

    # calculate window function and its fft
    print "Setting up window function and its fft..."
    start = time.time()
    # check for window function directory
    wdir = "./windows"
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    wfft_filename = wdir+"/wfft"+wparams
    # if file doesnt exist, calculate the fft, but if it does, just load file
    if os.path.isfile(wfft_filename):
        W_fft = np.load(wfft_filename)
    else:
        W_grid = window_gauss_sigmoid(x_grid, y_grid, z_grid, params)
        # normalize window function
        W_grid_norm = W_grid/np.sum(W_grid)
        # fft the window function
        W_fft = np.fft.rfftn(W_grid_norm)
        # now lets save the W_fft and just load it everytime we need it, so we dont calculate this for all the sims
        np.save(wfft_filename, W_fft)

    print "Time elapsed: ", time.time() - start
    
    # convolve the post tsc grid with the window function
    # should make separate methods for gals and pairs
    gal_grid_convolved, gal_fft = gal_convolve_W(gal_grid, W_fft, params)

    print "Calculating galaxy bias..."
    # calculating overdensity fields
    rho_gal_avg = np.mean(gal_grid_convolved)
    delta_gal = (gal_grid_convolved - rho_gal_avg)/rho_gal_avg

    # weighted average of the galaxy overdensity field
    b_gals = np.sum(delta_gal*gal_grid)/np.sum(gal_grid)
    print "b_gals = ", b_gals

    # calculate pair bias
    # create bins for dist_los and dist_trans
    bins_los = np.linspace(0,params['maxdist']/params['Lbox'],num=dist_nbins + 1)
    bins_trans = np.linspace(0,params['maxdist']/params['Lbox']/3,num=dist_nbins + 1)
    # open a file to store bias results
    filename = savedir+"/data"+wtype+"_"+str(dist_nbins)+"_"+str(dist_nbins)+"_ilos_itrans_bpair"+wparams+"_maxdist"+str(int(params['maxdist']))+"_sim"+str(whichsim)+".txt"
    f = open(filename,'w')
    f.write("%s \n" % (b_gals))
    f.close()

    # for each ilos, loop through itrans to fill in bs_pair
    for ilos in range(0, dist_nbins):
        for itrans in range(0, dist_nbins):
            # find the indices of all the pairs that fall within this dist bin
            masks_los = [dist_los > bins_los[ilos], dist_los < bins_los[ilos+1]]
            masks_trans = [dist_trans > bins_trans[itrans], dist_trans < bins_trans[itrans+1]]
            allmasks = masks_los + masks_trans
            tot_mask = reduce(np.logical_and, allmasks)
            tot_mask = np.array(tot_mask)
            # if the sub sample is empty, skip
            if np.sum(tot_mask) == 0:
                continue
            # pull out the subsample of pairs
            pos_pairs_sub = pos_pairs[tot_mask]
            dist_pairs_sub = dist_pairs[tot_mask]
            gal_ids_sub = gal_ids[tot_mask]

            # calculating average overdensity of pairs, averaged over pair positions
            # calculate the list of odens at all pairs
            odens_all_pairs = np.zeros(len(pos_pairs_sub))
            for i in range(0, len(pos_pairs_sub)):
                odens_all_pairs[i] = interp_odens(pos_pairs_sub[i], delta_gal, params)
            # average b_pair averaged over all pairs
            b_pair = np.mean(odens_all_pairs)
            # output
            if b_pair/b_gals < 1:
                print "bin: ", ilos, itrans, ", b_pair = ", b_pair, ", Number of pairs: ", np.sum(tot_mask)
            f = open(filename, 'a')
            f.write("%s %s %s %s \n" % (ilos, itrans, b_pair, np.sum(tot_mask)))
            f.close()




