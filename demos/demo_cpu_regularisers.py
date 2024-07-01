#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:39:43 2018

Demonstration of CPU regularisers 

@authors: Daniil Kazantsev, Edoardo Pasca
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import timeit
from imageio.v2 import imread

from ccpi.filters.regularisers import ROF_TV, FGP_TV, PD_TV, SB_TV, TGV, LLT_ROF, FGP_dTV, TNV, NDF, Diff4th
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
print ("_______________ROF-TV (2D)_________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of ROF-TV regulariser using the CPU')
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

info_vec = np.zeros(2, dtype = np.float32)
print ("#############ROF TV CPU####################")
start_time = timeit.default_timer()
rof_cpu = ROF_TV(pars['input'],
             pars['regularisation_parameter'],
             pars['number_of_iterations'],
             pars['time_marching_parameter'],
             pars['tolerance_constant'], device = 'cpu', infovector=info_vec)

Qtools = QualityTools(Im, rof_cpu)
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
imgplot = plt.imshow(rof_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________FGP-TV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of FGP-TV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : FGP_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :1500 ,\
        'tolerance_constant':1e-08,\
        'methodTV': 0 ,\
        'nonneg': 0}
        
print ("#############FGP TV CPU####################")
start_time = timeit.default_timer()
fgp_cpu = FGP_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'], device ='cpu')

Qtools = QualityTools(Im, fgp_cpu)
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
imgplot = plt.imshow(fgp_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________PD-TV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of PD-TV regulariser using the CPU')
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
        
print ("#############PD TV CPU####################")
start_time = timeit.default_timer()
pd_cpu = PD_TV(pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
        )

Qtools = QualityTools(Im, pd_cpu)
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
imgplot = plt.imshow(pd_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_______________SB-TV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of SB-TV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : SB_TV, \
        'input' : u0,\
        'regularisation_parameter':0.02, \
        'number_of_iterations' :100 ,\
        'tolerance_constant':1e-06,\
        'methodTV': 0}
        
print ("#############SB TV CPU####################")
start_time = timeit.default_timer()
sb_cpu = SB_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'], device='cpu')
             
#Qtools = QualityTools(Im, sb_cpu)
#pars['rmse'] = Qtools.rmse()

txtstr = printParametersToString(pars)
txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
a=fig.add_subplot(1,2,2)

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
# place a text box in upper left in axes coords
a.text(0.15, 0.25, txtstr, transform=a.transAxes, fontsize=14,
         verticalalignment='top', bbox=props)
imgplot = plt.imshow(sb_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))
#%%

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("______________LLT- ROF (2D)________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of LLT-ROF regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

# set parameters
pars = {'algorithm' : LLT_ROF, \
        'input' : u0,\
        'regularisation_parameterROF':0.01, \
        'regularisation_parameterLLT':0.0085, \
        'number_of_iterations' :6000 ,\
        'time_marching_parameter' :0.001 ,\
        'tolerance_constant':1e-06}
        
print ("#############LLT- ROF CPU####################")
start_time = timeit.default_timer()
lltrof_cpu = LLT_ROF(pars['input'], 
              pars['regularisation_parameterROF'],
              pars['regularisation_parameterLLT'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'], device = 'cpu')

Qtools = QualityTools(Im, lltrof_cpu)
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
imgplot = plt.imshow(lltrof_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("_____Total Generalised Variation (2D)______")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of TGV regulariser using the CPU')
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

print ("#############TGV CPU####################")
start_time = timeit.default_timer()
(tgv_cpu,info_vec_cpu)  = TGV(pars['input'], 
              pars['regularisation_parameter'],
              pars['alpha1'],
              pars['alpha0'],
              pars['number_of_iterations'],
              pars['LipshitzConstant'],
              pars['tolerance_constant'], 'cpu')

Qtools = QualityTools(Im, tgv_cpu)
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
imgplot = plt.imshow(tgv_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("________________NDF (2D)___________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of NDF regulariser using the CPU')
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
        
print ("#############NDF CPU################")
start_time = timeit.default_timer()
(ndf_cpu,info_vec_cpu) = NDF(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],
              pars['tolerance_constant'],'cpu')
             
Qtools = QualityTools(Im, ndf_cpu)
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
imgplot = plt.imshow(ndf_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("___Anisotropic Diffusion 4th Order (2D)____")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of Diff4th regulariser using the CPU')
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
        
print ("#############Diff4th CPU################")
start_time = timeit.default_timer()
(diff4_cpu,info_vec_cpu) = Diff4th(pars['input'], 
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'],
              pars['tolerance_constant'],'cpu')
             
Qtools = QualityTools(Im, diff4_cpu)
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
imgplot = plt.imshow(diff4_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))

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
              pars['edge_parameter'],'cpu')
              
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
plt.suptitle('Performance of NLTV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

pars2 = {'algorithm' : NLTV, \
        'input' : u0,\
        'H_i': H_i, \
        'H_j': H_j,\
        'H_k' : 0,\
        'Weights' : Weights,\
        'regularisation_parameter': 0.04,\
        'iterations': 3
        }
start_time = timeit.default_timer()
nltv_cpu = NLTV(pars2['input'], 
              pars2['H_i'],
              pars2['H_j'], 
              pars2['H_k'],
              pars2['Weights'],
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
print ("_____________FGP-dTV (2D)__________________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of FGP-dTV regulariser using the CPU')
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
        
print ("#############FGP dTV CPU####################")
start_time = timeit.default_timer()
(fgp_dtv_cpu,info_vec_cpu) = FGP_dTV(pars['input'], 
              pars['refdata'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['eta_const'], 
              pars['methodTV'],
              pars['nonneg'],'cpu')
             
Qtools = QualityTools(Im, fgp_dtv_cpu)
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
imgplot = plt.imshow(fgp_dtv_cpu, cmap="gray")
plt.title('{}'.format('CPU results'))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("__________Total nuclear Variation__________")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

## plot 
fig = plt.figure()
plt.suptitle('Performance of TNV regulariser using the CPU')
a=fig.add_subplot(1,2,1)
a.set_title('Noisy Image')
imgplot = plt.imshow(u0,cmap="gray")

channelsNo = 5
noisyVol = np.zeros((channelsNo,N,M),dtype='float32')
idealVol = np.zeros((channelsNo,N,M),dtype='float32')

for i in range (channelsNo):
    noisyVol[i,:,:] = Im + np.random.normal(loc = 0 , scale = perc * Im , size = np.shape(Im))
    idealVol[i,:,:] = Im

# set parameters
pars = {'algorithm' : TNV, \
        'input' : noisyVol,\
        'regularisation_parameter': 0.04, \
        'number_of_iterations' : 200 ,\
        'tolerance_constant':1e-05
        }
        
print ("#############TNV CPU#################")
start_time = timeit.default_timer()
tnv_cpu = TNV(pars['input'],
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'])

Qtools = QualityTools(idealVol, tnv_cpu)
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
imgplot = plt.imshow(tnv_cpu[3,:,:], cmap="gray")
plt.title('{}'.format('CPU results'))
