#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of GPU regularisers using CuPy wrappers
"""

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import os
import timeit
from imageio.v2 import imread

from ccpi.filters.regularisersCuPy import ROF_TV as ROF_TV_cupy
from ccpi.filters.regularisersCuPy import PD_TV as PD_TV_cupy

from ccpi.supp.qualitymetrics import QualityTools
###############################################################################
def printParametersToString(pars):
        txt = r''
        for key, value in pars.items():
            if key== 'algorithm' :
                txt += "{0} = {1}".format(key, value.__name__)
            elif key == 'input':
                txt += "{0} = {1}".format(key, np.shape(value))
            elif key == 'refdata':
                txt += "{0} = {1}".format(key, np.shape(value))
            else:
                txt += "{0} = {1}".format(key, value)
            txt += '\n'
        return txt
###############################################################################
filename = os.path.join( "../test/test_data" ,"peppers.tif")

# read image
Im = imread(filename)

Im = Im/255.0
perc = 0.05
u0 = Im + np.random.normal(loc = 0 ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
u_ref = Im + np.random.normal(loc = 0 ,
                                  scale = 0.01 * Im , 
                                  size = np.shape(Im))
(N,M) = np.shape(u0)
u0 = u0.astype('float32')
u_ref = u_ref.astype('float32')

slices = 20

noisyVol = np.zeros((slices,N,M),dtype='float32')
noisyRef = np.zeros((slices,N,M),dtype='float32')
idealVol = np.zeros((slices,N,M),dtype='float32')

for i in range (slices):
    noisyVol[i,:,:] = Im + np.random.normal(loc = 0 , scale = perc * Im , size = np.shape(Im))
    noisyRef[i,:,:] = Im + np.random.normal(loc = 0 , scale = 0.01 * Im , size = np.shape(Im))
    idealVol[i,:,:] = Im
    
noisyVol = np.float32(noisyVol)

# move numpy array to CuPy. 
# NOTE: Here we also need to be specific to which device we move the data to
gpu_device = 0 # select the device (if many)
with cp.cuda.Device(gpu_device):
    noisyVol_cp = cp.asarray(noisyVol, order="C")
#%%
# print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print ("_______________ROF-TV (3D)_________________")
# print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of ROF-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy 15th slice of the volume')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm': ROF_TV_cupy}

start_time = timeit.default_timer()

with cp.cuda.Device(gpu_device):
    rof_gpu3D = ROF_TV_cupy(noisyVol_cp,
                    regularisation_parameter=0.02,
                    iterations = 4000,
                    time_marching_parameter=0.001,
                    gpu_id=gpu_device)

    rof_gpu3D = rof_gpu3D.get() # back to numpy

Qtools = QualityTools(idealVol, rof_gpu3D)
pars['rmse'] = Qtools.rmse()
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(rof_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using ROF-TV'))
plt.savefig("rof_tv_image.png", dpi=250)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________PD-TV (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot 
fig = plt.figure()
plt.suptitle('Performance of PD-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy 15th slice of the volume')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm': PD_TV_cupy}

start_time = timeit.default_timer()

with cp.cuda.Device(gpu_device):
    pdtv_gpu3D = PD_TV_cupy(noisyVol_cp,
                    regularisation_parameter=0.06,
                    iterations = 2000,
                    methodTV=0,
                    nonneg=0,
                    lipschitz_const=8,
                    gpu_id=gpu_device)

    pdtv_gpu3D = pdtv_gpu3D.get() # back to numpy

Qtools = QualityTools(idealVol, pdtv_gpu3D)
pars['rmse'] = Qtools.rmse()
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(pdtv_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using PD-TV'))
plt.savefig("pd_tv_image.png", dpi=250)