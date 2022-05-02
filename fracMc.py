#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 16:56:57 2022

@author: ckervazo
"""
import numpy as np
import cv2
from scipy.interpolate import griddata

#%%
def fracMc(ref,mvf,outofbound=16):
    
    [rows, cols] = np.shape(ref);
    
    
    
    mc_c, mc_r = np.meshgrid(np.arange(cols),np.arange(rows)) # Initial coordinates, used to infer the new coordinates
    
    mc_r_tmp = mc_r - mvf[:,:,0]  # Row coordinates, after the movement
    mc_c_tmp = mc_c - mvf[:,:,1] # Column coordinates, after the movement
    
    extension = outofbound
    ref = cv2.copyMakeBorder(ref, extension, extension, extension, extension, cv2.BORDER_REFLECT)# Padded image
    [rows1, cols1] = np.shape(ref);
    mc_r = mc_r_tmp + extension # Row coordinates, after the movement (in the padded image)
    mc_c = mc_c_tmp + extension # Column coordinates, after the movement (in the padded image)
    
    colMeshGrid, rowMeshGrid = np.meshgrid(np.arange(cols1), np.arange(rows1)) # Original coordinates, in the padded image
    
    
    # Put the input coordinates into the form of an array, to mach scipy format
    points = np.zeros((rows1*cols1,2))
    points[:,0] = colMeshGrid.reshape(rows1*cols1)
    points[:,1] = rowMeshGrid.reshape(rows1*cols1)
    
    values = ref.reshape(rows1*cols1)    
    
    
    # Put the coordinates in which we want ot do the interpolation into the form of an array, to mach scipy format
    points_inter = np.zeros((rows*cols,2))
    points_inter[:,0] = mc_c.reshape(rows*cols)
    points_inter[:,1] = mc_r.reshape(rows*cols)
    
    motcomp =  griddata(points, values, points_inter, method='linear')# Perform the interpolation after the movement
    
    motcomp = motcomp.reshape(rows,cols)

    return motcomp