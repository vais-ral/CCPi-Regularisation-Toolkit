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
from ccpi.filters.regularisers import PatchSelect, NLTV
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
perc = 0.08
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

plt.figure()
plt.imshow(u0, cmap="gray")
plt.show()
#%%

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________ROF-TV regulariser_____________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of the ROF-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm': ROF_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02,\
        'number_of_iterations': 4000,\
        'time_marching_parameter': 0.001,\
        'tolerance_constant':1e-06}

print ("##############ROF TV GPU##################")
start_time = timeit.default_timer()
info_vec_gpu = np.zeros(2, dtype = np.float32)
rof_gpu = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],
             pars['tolerance_constant'], device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(Im, rof_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = ROF_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(rof_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________FGP-TV regulariser_____________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of the FGP-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :400 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0 ,\
        'nonneg': 0}

print ("##############FGP TV GPU##################")
start_time = timeit.default_timer()
fgp_gpu= FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],device='gpu', infovector=info_vec_gpu)
Qtools = QualityTools(Im, fgp_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = FGP_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________PD-TV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of PD-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : PD_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :1500 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0 ,\
        'nonneg': 1,
        'lipschitz_const' : 8}
        
print ("#############PD TV GPU####################")
start_time = timeit.default_timer()
pd_gpu= PD_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'],
              pars['lipschitz_const'],
              pars['methodTV'],
              pars['nonneg'],
              device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(Im, pd_gpu)
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
imgplot = plt.imshow(pd_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________SB-TV regulariser______________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of the SB-TV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : SB_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :250 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0}

print ("##############SB TV GPU##################")
start_time = timeit.default_timer()
sb_gpu = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'], device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(Im, sb_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = SB_TV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(sb_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))
#%%

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("______________LLT- ROF (2D)________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of LLT-ROF regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : LLT_ROF, \
        'input' : u0,\
        'regularisation_parameterROF':0.01, \
        'regularisation_parameterLLT':0.0085, \
        'number_of_iterations' : 6000 ,\
        'time_marching_parameter' :0.001 ,\
        'tolerance_constant':1e-06}
        
print ("#############LLT- ROF GPU####################")
start_time = timeit.default_timer()
lltrof_gpu = LLT_ROF(pars['input'], 
              pars['regularisation_parameterROF'],
              pars['regularisation_parameterLLT'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'], device='gpu', infovector=info_vec_gpu)
             
Qtools = QualityTools(Im, lltrof_gpu)
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
imgplot = plt.imshow(lltrof_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

#%%

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_____Total Generalised Variation (2D)______")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of TGV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : TGV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'alpha1':1.0,\
        'alpha0':2.0,\
        'number_of_iterations' :1000 ,\
        'LipshitzConstant' :12 ,\
        'tolerance_constant':1e-06}
        
print ("#############TGV GPU####################")
start_time = timeit.default_timer()
tgv_gpu = TGV(pars['input'], 
              pars['regularisation_parameter'],
              pars['alpha1'],
              pars['alpha0'],
              pars['number_of_iterations'],
              pars['LipshitzConstant'],
              pars['tolerance_constant'],device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(Im, tgv_gpu)
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
imgplot = plt.imshow(tgv_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________NDF regulariser_____________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of the NDF regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : NDF, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'edge_parameter':0.017,\
        'number_of_iterations' :1500 ,\
        'time_marching_parameter':0.01,\
        'penalty_type':1,\
        'tolerance_constant':1e-06}

print ("##############NDF GPU##################")
start_time = timeit.default_timer()
ndf_gpu = NDF(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],
              pars['tolerance_constant'],device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(Im, ndf_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = NDF
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(ndf_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Anisotropic Diffusion 4th Order (2D)____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of Diff4th regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : Diff4th, \
        'input' : u0,\
        'regularisation_parameter':0.8, \
        'edge_parameter':0.02,\
        'number_of_iterations' :5500 ,\
        'time_marching_parameter':0.001,\
        'tolerance_constant':1e-06}
        
print ("#############DIFF4th GPU################")
start_time = timeit.default_timer()
diff4_gpu = Diff4th(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'],device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(Im, diff4_gpu)
pars['algorithm'] = Diff4th
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
imgplot = plt.imshow(diff4_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Nonlocal patches pre-calculation____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
start_time = timeit.default_timer()
# set parameters
pars = {'algorithm' : PatchSelect, \
        'input' : u0,\
        'searchwindow': 7, \
        'patchwindow': 2,\
        'neighbours' : 15 ,\
        'edge_parameter':0.18}

H_i, H_j, Weights = PatchSelect(pars['input'], 
              pars['searchwindow'],
              pars['patchwindow'], 
              pars['neighbours'],
              pars['edge_parameter'],device='gpu')
              
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
"""
plt.figure()
plt.imshow(Weights[0,:,:],cmap="gray",interpolation="nearest",vmin=0, vmax=1)
plt.show()
"""
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Nonlocal Total Variation penalty____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
## plot 
fig = plt.figure()
plt.suptitle('Performance of NLTV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

pars2 = {'algorithm' : NLTV, \
        'input' : u0,\
        'H_i': H_i, \
        'H_j': H_j,\
        'H_k' : 0,\
        'Weights' : Weights,\
        'regularisation_parameter': 0.02,\
        'iterations': 3,
        'neighbours': 15,
        }
start_time = timeit.default_timer()
nltv_cpu = NLTV(pars2['input'], 
              pars2['H_i'],
              pars2['H_j'], 
              pars2['H_k'],
              pars2['Weights'],
              pars2['neighbours'],
              pars2['regularisation_parameter'],
              pars2['iterations'])

Qtools = QualityTools(Im, nltv_cpu)
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
imgplot = plt.imshow(nltv_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("____________FGP-dTV bench___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of the FGP-dTV regulariser using the GPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_dTV, \
        'input' : u0,\
        'refdata' : u_ref,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :500 ,\
        'tolerance_constant':1e-06,\
        'eta_const':0.2,\
        'methodTV': 0 ,\
        'nonneg': 0}

print ("##############FGP dTV GPU##################")
start_time = timeit.default_timer()
fgp_dtv_gpu = FGP_dTV(pars['input'], 
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'], 
              pars['methodTV'],
              pars['nonneg'],device='gpu', infovector=info_vec_gpu)

Qtools = QualityTools(Im, fgp_dtv_gpu)
pars['rmse'] = Qtools.rmse()
pars['algorithm'] = FGP_dTV
txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(fgp_dtv_gpu, cmap="gray")
plt.title('{}'.format('GPU results'))
#%%