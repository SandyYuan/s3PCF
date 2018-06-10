#!/usr/bin/env python

import numpy as np
import os,sys

from s3PCF import calc_pair_bias as calcbias

# constants
params = {}
params['z'] = 0.5
params['h'] = 0.6726
params['Nslab'] = 3
params['Lbox'] = 1100/params['h'] # Mpc, box size
params['Mpart'] = 3.88537e+10/params['h'] # Msun, mass of each particle
params['velz2kms'] = 9.690310687246482e+04/params['Lbox'] # H(z)/(1+Z), km/s/Mpc
params['maxdist'] = 30 # Mpc # use 10 Mpc for the real space case
params['num_sims'] = 16

# the standard HOD, Zheng+2009, Kwan+2015
M_cut = 10**13.35 # these constants are taken at the middle of the design, Kwan+15
log_Mcut = np.log10(M_cut)
M1 = 10**13.8
log_M1 = np.log10(M1)
sigma = 0.85
alpha = 1
kappa = 1
A = 0

# rsd?
rsd = True
params['rsd'] = rsd

whichsim = 0

# the data directory 
datadir = "../gal_profile/data"
if rsd:
    datadir = datadir+"_rsd"
savedir = datadir+"/rockstar_"+str(M_cut)[0:4]+"_"+str(M1)[0:4]+"_"+str(sigma)[0:4]+"_"+str(alpha)[0:4]+"_"+str(kappa)[0:4]+"_"+str(A)
if rsd:
    savedir = savedir+"_rsd"

# load the galaxy and pair data
print "Loading pair/galaxy catalogs...", whichsim
pos_full = np.fromfile(savedir+"/halos_gal_full_pos_"+str(whichsim))
pair_data = np.fromfile(savedir+"/halos_pairs_full_"+str(whichsim)+"_maxdist"+str(int(params['maxdist'])))
pos_full = np.array(np.reshape(pos_full, (-1, 3))) / params['Lbox'] - 0.5 # relative unit
pair_data = np.array(np.reshape(pair_data, (-1, 7)))

# compute pair bias and Qeff
calcbias.calc_qeff(whichsim, pos_full, pair_data, params, rsd = params['rsd'])

