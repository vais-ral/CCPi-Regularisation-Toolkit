#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Demonstration of GPU regularisers

@authors: Daniil Kazantsev, Edoardo Pasca
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from imageio.v2 import imread


from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, NDF, Diff4th
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
#%%
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
# map the u0 u0->u0>0
# f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
u0 = u0.astype('float32')
u_ref = u_ref.astype('float32')

slices = 20

noisyVol = np.zeros((slices,N,N),dtype='float32')
noisyRef = np.zeros((slices,N,N),dtype='float32')
idealVol = np.zeros((slices,N,N),dtype='float32')

for i in range (slices):
    noisyVol[i,:,:] = Im + np.random.normal(loc = 0 , scale = perc * Im , size = np.shape(Im))
    noisyRef[i,:,:] = Im + np.random.normal(loc = 0 , scale = 0.01 * Im , size = np.shape(Im))
    idealVol[i,:,:] = Im

info_vec_gpu = np.zeros((2,), dtype='float32')
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________ROF-TV (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of ROF-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy 15th slice of a volume')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm': ROF_TV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.02,\
        'number_of_iterations': 7000,\
        'time_marching_parameter': 0.0007,\
        'tolerance_constant':1e-06}

print ("#############ROF TV GPU####################")
start_time = timeit.default_timer()
rof_gpu3D = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],
              pars['tolerance_constant'], device='gpu', infovector=info_vec_gpu)

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
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________FGP-TV (3D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of FGP-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : FGP_TV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :1000 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0 ,\
        'nonneg': 0}

print ("#############FGP TV GPU####################")
start_time = timeit.default_timer()
fgp_gpu3D  = FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'], device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(idealVol, fgp_gpu3D)
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
imgplot = plt.imshow(fgp_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using FGP-TV'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________PD-TV (3D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of PD-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : PD_TV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :1000 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0 ,\
        'nonneg': 0,
        'lipschitz_const' : 8}

print ("#############PD TV GPU####################")
start_time = timeit.default_timer()
pd_gpu3D  = PD_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['lipschitz_const'],
              pars['methodTV'],
              pars['nonneg'],
              device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(idealVol, pd_gpu3D)
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
imgplot = plt.imshow(pd_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using PD-TV'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________SB-TV (3D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of SB-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : SB_TV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :300 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0 }

print ("#############SB TV GPU####################")
start_time = timeit.default_timer()
sb_gpu3D = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(idealVol, sb_gpu3D)
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
imgplot = plt.imshow(sb_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using SB-TV'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________LLT-ROF (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of LLT-ROF regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : LLT_ROF, \
        'input' : noisyVol,\
        'regularisation_parameterROF':0.01, \
        'regularisation_parameterLLT':0.008, \
        'number_of_iterations' : 500 ,\
        'time_marching_parameter' :0.001 ,\
        'tolerance_constant':1e-06}

print ("#############LLT ROF GPU####################")
start_time = timeit.default_timer()
lltrof_gpu3D = LLT_ROF(pars['input'], 
              pars['regularisation_parameterROF'],
              pars['regularisation_parameterLLT'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'], device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(idealVol, lltrof_gpu3D)
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
imgplot = plt.imshow(lltrof_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using LLT-ROF'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________TGV (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of TGV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : TGV, \
        'input' : noisyVol,\
        'regularisation_parameter':0.02, \
        'alpha1':1.0,\
        'alpha0':2.0,\
        'number_of_iterations' :500 ,\
        'LipshitzConstant' :12 ,\
        'tolerance_constant':1e-06}

print ("#############TGV GPU####################")
start_time = timeit.default_timer()
tgv_gpu3D   = TGV(pars['input'], 
              pars['regularisation_parameter'],
              pars['alpha1'],
              pars['alpha0'],
              pars['number_of_iterations'],
              pars['LipshitzConstant'],
              pars['tolerance_constant'], device='gpu', infovector=info_vec_gpu)


Qtools = QualityTools(idealVol, tgv_gpu3D)
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
imgplot = plt.imshow(tgv_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using TGV'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________NDF-TV (3D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of NDF regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : NDF, \
        'input' : noisyVol,\
        'regularisation_parameter':0.02, \
        'edge_parameter':0.015,\
        'number_of_iterations' :700 ,\
        'time_marching_parameter':0.01,\
        'penalty_type':  1,\
        'tolerance_constant':1e-06}


print ("#############NDF GPU####################")
start_time = timeit.default_timer()
ndf_gpu3D  = NDF(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],
              pars['tolerance_constant'], device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(idealVol, ndf_gpu3D)
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
imgplot = plt.imshow(ndf_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using NDF'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Anisotropic Diffusion 4th Order (3D)____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of DIFF4th regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : Diff4th, \
        'input' : noisyVol,\
        'regularisation_parameter':0.8, \
        'edge_parameter':0.02,\
        'number_of_iterations' :500 ,\
        'time_marching_parameter':0.001,\
        'tolerance_constant':1e-06}
        
print ("#############DIFF4th GPU################")
start_time = timeit.default_timer()
diff4_gpu3D = Diff4th(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'],device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(idealVol, diff4_gpu3D)
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
imgplot = plt.imshow(diff4_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('GPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________FGP-dTV (3D)________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of FGP-dTV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(noisyVol[10,:,:],cmap="gray")

# set parameters
pars = {'algorithm' : FGP_dTV,\
        'input' : noisyVol,\
        'refdata' : noisyRef,\
        'regularisation_parameter':0.02,
        'number_of_iterations' :500 ,\
        'tolerance_constant':1e-06,\
        'eta_const':0.2,\
        'methodTV': 0 ,\
        'nonneg': 0}

print ("#############FGP TV GPU####################")
start_time = timeit.default_timer()
fgp_dTV_gpu3D  = FGP_dTV(pars['input'],
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'],
              pars['methodTV'],
              pars['nonneg'],device='gpu', infovector=info_vec_gpu)
             

Qtools = QualityTools(idealVol, fgp_dTV_gpu3D)
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
imgplot = plt.imshow(fgp_dTV_gpu3D[10,:,:], cmap="gray")
plt.title('{}'.format('Recovered volume on the GPU using FGP-dTV'))
#%%
