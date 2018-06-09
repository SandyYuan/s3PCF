#!/usr/bin/env python

#
# PowerSpectrum_mod.py
#
# The library interface to compute power spectra with a variety of options
# from a variety of file formats.
#
# For a user-friendly command-line interface, see `run_PS.py`.
#
#Created on: Mar 2, 2013
#     Author: dferrer
#     Updated 2016: lgarrison
#     Mod: syuan
import sys
import ctypes as ct
import numpy as np
np.seterr(all="warn")
import scipy
import scipy.interpolate
import os
import glob
from itertools import izip
from Abacus.Analysis.PowerSpectrum import PowerSpectrum as PS

from Abacus import Tools
from Abacus import ReadAbacus
#from loom import Histogram

pslibf = ct.cdll.LoadLibrary(os.getenv('ABACUS') + "/Analysis/PowerSpectrum/libpowerf.so")
pslibd = ct.cdll.LoadLibrary(os.getenv('ABACUS') + "/Analysis/PowerSpectrum/libpowerd.so")
pslibf.do_tsc_on_file_rvdouble.restype = ct.c_uint64
pslibf.do_tsc_on_file_pack14.restype = ct.c_uint64
pslibd.do_tsc_on_file_rvdouble.restype = ct.c_uint64
pslibd.do_tsc_on_file_pack14.restype = ct.c_uint64
pi = np.pi

verbose = True

def CalculateTSC(positions, gridsize, boxsize, weights=None, projected=False, dtype=np.float32, rotate_to=None, nbins=-1, log=False):
    """
    Computes the power spectrum of a set of points by binning them into a density field.
    
    Parameters
    ----------
    positions: ndarray of shape (N,3), or list of ndarray
        The points to bin
    gridsize: int
        The FFT mesh side length
    boxsize: float
        The physical domain size.
        `positions` are assumed to be in a zero-centered box of side length `boxsize`.
    weights: ndarray of shape (N,), or list of ndarray, optional
        Binning weights for the points.
    projected: bool, optional
        Whether to project the points along the z-axis
    dtype: np.dtype, optional
        The data type to do the computations in.
    rotate_to: array of floats of shape (3,), optional
        `rotate_to` defines a vector that the cartesian z-hat will be rotated to align with.
        All rotations are periodic "safe", meaning we discard particles at the edges.
        Default: None.
    projected: bool, optional
        Do 2D binning and FFT instead of 3D.  This is considerably more efficient.
        All projections are done along the z-axis after any rotation.
        Default: False.
    nbins: int, optional
        Number of k-space bins.  Default: gridsize/4.
    log: bool, optional
        Whether to bin in log space.  Default: False.
    
    Returns
    -------
    k: ndarray
        Average wavenumber per k-bin
    s: ndarray
        Average power per k-bin
    nb: ndarray
        Total modes per k-bin
    """
    
    # Pick the right library
    if dtype == np.float32:
        lib = pslibf
    elif dtype == np.float64:
        lib = pslibd
    else:
        raise ValueError(dtype, "Error! unsupported data type")
    
    if type(positions) != list:
        positions = [positions]
    if weights == None:
        weights = [None]*len(positions)
    if type(weights) != list:
        weights = [weights]
    assert len(positions) == len(weights)
        
    positions = [np.ascontiguousarray(p.reshape(-1,3), dtype=dtype) for p in positions]
    NPs = [len(p) for p in positions]
    if weights[0] != None:
        weights = [np.ascontiguousarray(w, dtype=dtype) for w in weights]
        assert NPs == [len(w) for w in weights]

    if projected:
        fielddensity = np.zeros((gridsize,gridsize), dtype=dtype)
    else:
        fielddensity = np.zeros((gridsize,gridsize,gridsize), dtype=dtype)
        
    if rotate_to:
        rotate_to = rotate_to.as_type(np.float64)

    if verbose:
        print 'Starting TSC...'
    # get the density weighted field grid
    for p,w,NP in izip(positions,weights,NPs):
        lib.do_tsc(Tools.cpointer(p), Tools.cpointer(w), Tools.cpointer(fielddensity), ct.c_uint64(NP), ct.c_uint64(gridsize), ct.c_double(boxsize), Tools.cpointer(rotate_to), ct.c_int(projected), ct.c_int(0))
    if verbose:
        print '\tDone.'
        #print 'Starting FFT and radial binning...'
    """
    # Do the FFT
    res = PS.FFTAndBin(fielddensity, boxsize, gridsize, twoD=projected, nbins=nbins, log=log, dtype=dtype, inplace=True)
    if verbose:
        print '\tDone'
    """
    return fielddensity


