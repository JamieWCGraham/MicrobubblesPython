#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:40:27 2019

@author: jamiegraham
"""

import numpy as np, numpy
import matplotlib as mlp
import matplotlib.pyplot as plt
from numpy import sqrt, abs, exp, int, cos, tan, sin, fft
from scipy.constants import c, h, k, pi, sigma, epsilon_0, e 
import random as random
import time
import scipy as scipy
from scipy import integrate
import scipy.io
import scipy.stats as st
from scipy import signal
import cv2 as cv2
from Polygon import PolygonDrawer
from sklearn.preprocessing import normalize
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import time
import matplotlib.collections as mcoll
from colorlines import multicolored_lines, colorline, make_segments
import dataframe as df
from sox import os 
from pylab import polyfit
duration = 1  # seconds
freq = 440  # Hz


""" This is the updated version of the velocity microbubble code. Apologies for the lack of 
    helpful commenting/general computational 'verbosity' in the MatLab code. This code is 
    written to be much more helpful and informative. 
    
    # TIPS FOR RUNNING CODE 
    
    1. The code takes a .mat file as input that holds the 288x354x1000 image data stored in a variable. This file is generated
    using a MATLAB code from before which I attached. That matlab code specifies the filename of the bmode.iq.mat file containing the image data.
    
    2. Run this code, wait for python to unpack the .mat file into a variable called xyt. (this will take a lil bit, the file is large)
    
    3. There are two ways to run this code, singlevesselmode and mode with multiple vessels. 
       Single vessel mode will analyze the image one vessel at a time, continually prompting the user for ROI input for the next vessel. 
       Multiple vessel mode will ask for all the ROI's to be inputted at the beginning, and then will just run the analysis without needing anymore input.
       I suggest using single vessel mode initially to get a feel for the program, and then using mult vessel mode once comfortable. 
    
    4. The code will ask you to input the number of vessels you want to analyze. input 1 for single vessel mode. input n (integer) for mult.
   
      The code will also ask you to input whether the vessel is an arteriole "A" or venule "V". It will also 
      ask for an intensity threshold. Look at the video that the MATLAB file generated to determine these two parameters. 
      For the intensity threshold, this helps the code filter out centroids that are most likely noise and not actually 
      microbubble signal. Vessels with very bright signal should have this parameter set ~= 50, vessels with not so bright signal ~= 45. 
      This needs to be played around with a bit to get the centroids accurate, and is why I suggest single vessel mode initially. 
   
    5. After input of these parameters, the code wants you to input two ROI's. the first ROI image will open up in a figure window, (may not pop up itself, look in bar)
       draw a rectangle ROI using mouse clicks that bounds the vessel (leaving some interior room for the second inner ROI). The function of this ROI is to essentially 
       crop the image so that we are not performing analysis on the whole image (would take a long time). When you are done, hit escape button, then hit any button to close window.
       
       Now, another window should open restricting the image to the rectangle ROI. Mouse click a polygon ROI (as many clicks as you like) around the vessel. 
       When you are done, hit escape button, then hit any other button to close window. 
     
    6. Wait for code to run, plots will start to pop up of various frames being analyzed. final plot is the 
       maximum intensity projection of the slice with the coloured velocity segments overlayed on the given vessel.
       

    
    PSEUDOCODE 
    
    # 1. Defining helpful functions
    # 2. Importing data from MATLAB generated files 
    # 3. Manually inputting our ROI's for individual vessels in the microbubble tracking. 
    # 4. Velocity Analysis on the centroid data across the 1000 frames.


"""


# 1. Defining helpful functions

" This function helps us highlight the centroids on our plots" 

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


" This function generates a Gaussian PSF of the system" 

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

    
""" This function localizes the microbubble centroids for a given frame t.
 There is a lot going on in this next function. Here's a roadmap. 

1. Initializing an array to store our microbubble centroid coordinates. 
2. Iterating over the x, and y axes of the image data. The idea for this double for loop
   is that it iterates across all the points on our 2D image array. For a given point (i,j), 
   suppose we consider a 3x3 pixel square centered at our point (i,j). Call this 3x3 square "datasquare"
3. Initializing the 3x3 matrix called 'datasquare'. 
4. These three lines of code serve to fill in the datasquare matrix. Each line defines a row of the matrix.
5. Setting boolean = True. As long as boolean = true, we continue to count the point (i,j) as a 
   microbubble centroid. If at any time in the following lines, boolean = False, then we throw away the point
   and start the for loop again for (i,j+1)
6. Here we have a for loop over l imbedded in a while loop over k, with boolean. 
   The idea for this portion of the code is to construct a set of pre-requisites 
   for the given coordinate (i,j) to be deemed a microbubble centroid. 
   These prerequisites are Logical1, Logical2, etc... 
   We iterate over k and l, and we compute Logical1 through Logical4, where
7. k is a var. that represents an index over the columns of datasquare
   l is a var. that represents an index over the rows of datasquare
8. Logical1 tests whether the point (i,j) has greater intensity than it's (k,l)'th neighbour.
            - iterating this over all k and l lets us know whether the point (i,j)
              is the brightest point in 'datasquare'. If the point (i,j) is the
              brightest point in a 3x3 grid around it, this is a characteristic property
              of the microbubble signal, and we set Logical1 = true 
   Logical2 tests whether the point (i,j) has intensity greater than a determined
            intensity threshold. We have set this to 52, but this may vary from 
            dataset to dataset. 
   Logical3 is our boolean logical from before.
   Logical4 neccessitates that the point (i,j) is in our polygon ROI that we 
            manually drew. 
9.  If all(Logical) is true, then we know (i,j) is a microbubble centroid.  
10. Otherwise... we set boolean = False, and then (i,j) is thrown out, and we look 
    to the next point (i,j+1)
    
11. If boolean is still true, (i,j) is a microbubble centroid
12. Increment the counter (total number of centroids)
13. Fill in our coordinates array with the coordinates (i,j) of our centroid.
14. If boolean is false, then (i,j) is not a microubble centroid, and we do nothing

15. This portion of the code just cleans up the array "coordinates" by 
    removing the zero elements in the array. It will likely be simplified 
    depending on final computational efficiency of the code. 

16. The function spits out 'coordinates2', an array containing microbubble centroid
    coordinates (y,x) for frame t. 

"""





def centroids(t,intensity_threshold,x1,y1,subset,Polygon_bucket,vesselnumber):
    coordinates = np.zeros((1000,2))   # 1. 
    counter = 0      
    for i in range(5,y1-5):             # 2. 
        for j in range(5,x1-5):
            
            polygon = Polygon_bucket[vesselnumber]
            if polygon.contains(Point(j,i)) == False:
                continue
            else:
                pass
            
            datasquare = np.zeros((3,3)) # 3.                      
            datasquare[0] = [subset[i-1][j-1][t],subset[i-1][j][t] ,subset[i-1][j+1][t]]    #4. 
            datasquare[1] = [subset[i][j-1][t],subset[i][j][t] ,subset[i][j+1][t]]
            datasquare[2] = [subset[i+1][j-1][t],subset[i+1][j][t] ,subset[i+1][j+1][t]]
            
            boolean = True      # 5. 
            k = -1
            l = 0
            while boolean and k<2:   # 6. 
                k+=1  # 7. 
                for l in range(3):
                    Logical1 = (datasquare[1][1] >= datasquare[k][l])  # 8. (iterated 9 times) 
                    Logical2 = (datasquare[1][1] > intensity_threshold)
                    Logical3 = boolean
                    Logical = [Logical1,Logical2,Logical3]
                    if all(Logical) == True:   # 9. 
                        pass
                    else:                      # 10. 
                        boolean = False

            
            if boolean == True:             # 11.
                counter+=1                  # 12. 
                coordinates[counter] = i,j  # 13. 
            else:    # 14. 
                pass
          


    length2 = len(coordinates[~np.all(coordinates==0,axis =1)])
    coordinates2 = np.zeros((100,2))
    coordinates2[0:length2] = coordinates[~np.all(coordinates==0,axis =1)]
              
    return coordinates2    # 16.


""" These functions simply plot the centroids on the image for a given frame """

def pltcentroids(centroids,t,color,subset):
    img = subset[:,:,t]
    plt.figure(figsize = (100,10))
    myfig = plt.imshow(img, cmap='ocean')
    
    for i in range(len(centroids)):
        highlight_cell(centroids[i][1],centroids[i][0], color="red", linewidth=2)
    
    plt.colorbar(label = "Intensity of Signal (W/m^2) ")
    plt.show()
    return img


def bubbleplot(t,subset,centroid_bucket):
    redsquares = centroid_bucket[t][~np.all(centroid_bucket[t] == 0, axis=1)]
    myfig = pltcentroids(redsquares,t,'red',subset)
    return myfig

    

# 2. Importing data from MATLAB file


if 'matimported' not in globals(): 
    a = scipy.io.loadmat('DD.mat')        # load .mat file containing the data
    print("keys are:", a.keys())          # it's stored in a dictionary, which we want to turn into a 3D matrix
    
    counter = 0                           # this is a counter that functions as a timer
    xyt = np.zeros((288,354,1000))        # initializing array that we will store our data in 
    n = 0                                 # just another counter 
    
    for i in range(288):                  # looping over x, y, t axes
        for j in range(354):
            for k in range(1000):
                   xyt[i][j][k] = a["DD"][i][j][k]   # iteratively storing data in new data structure
                   
                   counter += 1
                   if counter/101952000 > n/100:      # this is just for timing the code
                       n += 1
                       print(counter/101952000 , "%")

    print("DONE")

 
    # We'd also like to retrieve the maximum intensity projection from the constructed video. 

    MIP = np.zeros((288,354))
    for i in range(288):                  # looping over x, y, t axes
        for j in range(354):
                MIP[i][j] = max(xyt[i][j])    # Constructing MIP 
    
    plt.figure()
    plt.imshow(MIP,cmap = 'copper')
    matimported = 1

else:
    pass 







# Starting velocity analysis 

# initialization of data structures

v_bucket = []               # list of lists containing velocities for each vessel, each sub-list holds the velocity values for each segment of the vessel (lowest y centroid to highest y centroid) for a given vessel.  
X_bucket = []               # list of lists containing x coordinates for the vessel segment centroids
Y_bucket = []               # list of lists containing y coordinates for the vessel segment centroids

Polygon_bucket = []         # list of Polygon ROI's for the analyzed vessels
Rectangle_bucket = []       # list of rectangular ROI's for the analyzed vessels. 
attemptvessel1 = int(input("How many vessels to attempt to analyze?: "))
attemptvessel = attemptvessel1 

variance_bucket = []       # list of lists containing variance estimates corresponding to the v_bucket
vesseltypebucket = []      # list of strings corresponding to each vessels direction "Venule,Arteriole"
intensity_threshold_bucket = [] # list of intensity thresholds corresponding to each vessel. 

vesselnumber = -1          # initializing vessel index, we start at -1 just since our loop is a while loop. and we want to start at 0 
datacollectionperiod = "not done"   # while loop boolean

xmax_bucket = []          #  holds the min and max coordinates for the Rectangle ROI
xmin_bucket = []          #
ymax_bucket = []          #
ymin_bucket = []          # 


" If the code breaks, just reset the analysis to the end of the last vessel by running this block of code"

# Polygon_bucket.pop()
# Rectangle_bucket.pop()
# vesseltypebucket.pop()
# X_bucket.pop()
# Y_bucket.pop()
# v_bucket.pop()
# intensity_threshold_bucket.pop()
# vesselnumber -= 1
# variance_bucket.pop()
# datacollectionperiod = "not done"


"Inputting our ROIs"


if attemptvessel == 1:
    singlevesselmode = True
else:
# initializing singlevesselmode 
    singlevesselmode = False



if singlevesselmode == True:
    pass
else:
    for q in range(attemptvessel):
        
        Image_1 = xyt[:,:,999]         
        plt.figure()
        plt.imshow(Image_1, cmap='ocean', interpolation='nearest')     
        
        image = MIP
        image = image/np.amax(image)    #  maps the intensity values in the data into (0,1) for polygon ROI construction 
                
        
        # 3. Defining our ROI's for the microbubble tracking. 
        
        # RECTANGLE ROI : restricts our image to a smaller image containing the vessel.
        
        vesseltype = str(input("Venule or Arteriole? (input V or A): "))
        intensity_threshold = int(input("Input Intensity Threshold:"))
     
            
        rectangle = PolygonDrawer("Polygon")   # We are going to restrict our analysis to a manually drawn ROI 
        image2 = rectangle.run(image)          # running Polygon constructor
        print("Polygon = %s" % rectangle.points)
        
        myrectangle = rectangle.points                 
        rectangle = Polygon(myrectangle)
        
        
        # POLYGON ROI : further restricts the smaller image to the polygon ROI 
        
        xvals = np.zeros(len(myrectangle))
        yvals = np.zeros(len(myrectangle))
        for i in range(len(myrectangle)):
           xvals[i] =  myrectangle[i][0]
           yvals[i] =  myrectangle[i][1]
        
        xmax = int(max(xvals) + 20)
        xmin = int(min(xvals) - 20)
        ymax = int(max(yvals) + 20)
        ymin = int(min(yvals) - 20)
        
    
        image = MIP[ymin:ymax,xmin:xmax]
        image2 = image/np.amax(image)    #  maps the intensity values in the data into (0,1) for polygon ROI construction 
    
        
        pd = PolygonDrawer("Polygon")   # We are going to restrict our analysis to a manually drawn ROI 
        image3 = pd.run(image2)          # running Polygon constructor
        print("Polygon = %s" % pd.points)
        
        mypolygon = pd.points                 
        polygon = Polygon(mypolygon)
        
        Rectangle_bucket.append(rectangle)   # storing our results for iteration 
        Polygon_bucket.append(polygon)
        xmax_bucket.append(xmax)
        ymax_bucket.append(ymax)
        ymin_bucket.append(ymin)
        xmin_bucket.append(xmin)
        vesseltypebucket.append(vesseltype)
        intensity_threshold_bucket.append(intensity_threshold)
    
        
        """             # x y     - note, this is opposite of the (col,row) convention we've been using 
        point = Point(23,32)            # testing Polygon booleans
        polygon.contains(point)         
        """
    

" Velocity Analysis "

while datacollectionperiod == "not done" :
    
    vesselnumber += 1
    
    if vesselnumber > attemptvessel-1 and singlevesselmode == False:
        datacollectionperiod == "done"
        vesselnumber -= 1
        break
    else:
        pass
    
    if singlevesselmode:
        
        
        vesseltype = str(input("Venule or Arteriole? (input V or A): "))
        intensity_threshold = int(input("Input Intensity Threshold:"))
     
        
        
        Image_1 = xyt[:,:,999]         
        plt.figure()
        plt.imshow(Image_1, cmap='ocean', interpolation='nearest')     
        
        image = MIP
        image = image/np.amax(image)    #  maps the intensity values in the data into (0,1) for polygon ROI construction 
                
            
        
        rectangle = PolygonDrawer("Polygon")   # We are going to restrict our analysis to a manually drawn ROI 
        image2 = rectangle.run(image)          # running Polygon constructor
        print("Polygon = %s" % rectangle.points)
        
        myrectangle = rectangle.points                 
        rectangle = Polygon(myrectangle)
        
        
        
        
        # POLYGON ROI : further restricts the smaller image to the polygon ROI 
        
        xvals = np.zeros(len(myrectangle))
        yvals = np.zeros(len(myrectangle))
        for i in range(len(myrectangle)):
           xvals[i] =  myrectangle[i][0]
           yvals[i] =  myrectangle[i][1]
        
        xmax = int(max(xvals) + 5)
        xmin = int(min(xvals) - 5)
        ymax = int(max(yvals) + 5)
        ymin = int(min(yvals) - 5)
        
         
        image = MIP[ymin:ymax,xmin:xmax]
        image2 = image/np.amax(image)    #  maps the intensity values in the data into (0,1) for polygon ROI construction 
                
            
        
        pd = PolygonDrawer("Polygon")   # We are going to restrict our analysis to a manually drawn ROI 
        image3 = pd.run(image2)          # running Polygon constructor
        print("Polygon = %s" % pd.points)
        
        mypolygon = pd.points                 
        polygon = Polygon(mypolygon)
        
        Rectangle_bucket.append(rectangle)   # storing our results for iteration 
        Polygon_bucket.append(polygon)
        xmax_bucket.append(xmax)
        ymax_bucket.append(ymax)
        ymin_bucket.append(ymin)
        xmin_bucket.append(xmin)
    
    else:
        pass
       
    
        
    print("Now analyzing vessel #", vesselnumber)
    
    t0 = time.time()
        
        
    
    subset = xyt[ymin_bucket[vesselnumber]:ymax_bucket[vesselnumber],xmin_bucket[vesselnumber]:xmax_bucket[vesselnumber],:]
    ytransform_min = ymin_bucket[vesselnumber]
    ytransform_max = ymax_bucket[vesselnumber]
    xtransform_min = xmin_bucket[vesselnumber]
    xtransform_max = xmax_bucket[vesselnumber]
    x1 = len(subset[0])
    y1 = len(subset)
    subset = subset[0:y1,0:x1]
    

    """ Here we are calculating all centroids in all frames in our ROI"""
    
    
    timer_bucket = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]     # pre-allocation 
    centroid_bucket = np.zeros((1000,100,2))
    
    
    print("Centroids being localized across 1000 frames: Frame number = ")
    for t in range(1000):
        centroid_bucket[t] = centroids(t,intensity_threshold,x1,y1,subset,Polygon_bucket,vesselnumber) 
        if t in timer_bucket:
            print(t)
        
    print("Done")
    
    
    """   
    # 5. Velocity Analysis on the centroid data across the 1000 frames.  
        
    """    
   
    "   # Search through centroid_bucket for microbubble spiking events "
    
    frame_indices = []
    
    for t in range(1000):
        centroid_count = len(centroid_bucket[t][~np.all(centroid_bucket[t] == 0, axis =1)])
        if centroid_count >= 4:
            frame_indices.append((t,centroid_count))


    """   
    
    # search through these selected frames now for max spiking events 
    """     
      
    frame_indices2 = []
    for i in range(1,len(frame_indices)-1):
       if frame_indices[i-1][1] + 1 == frame_indices[i][1] == frame_indices[i+1][1] == frame_indices[i+2][1] == frame_indices[i+3][1]:
           frame_indices2.append(frame_indices[i])
       else:
           pass

   
    # "   # plot such events "
            
    # for i in range(len(frame_indices2)):
    #     bubbleplot(frame_indices2[i][0],subset,centroid_bucket)
        
        
    "   # compute Euclidean distances between the microbubble centroids. "
    
    temp_bucket = [] # holds the microbubble centroids we want to analyze
    for i in range(len(frame_indices2)):
        temp_bucket.append(centroid_bucket[frame_indices2[i][0]][~np.all(centroid_bucket[frame_indices2[i][0]] == 0 , axis = 1)])
    
    
    "   # filtering out non-linear centroid frames that constitute noise"
    removelist = []
    for i in range(len(temp_bucket)):   # iterating over our centroid bucket
        slopes = []
        for j in range(len(temp_bucket[i])-1): # iterating over the centroids in a given frame
            temp_1 = temp_bucket[i][j]
            temp_2 = temp_bucket[i][j+1]
            if temp_2[1] - temp_1[1] != 0:
                slope = (temp_1[0] - temp_2[0]) / (temp_1[1] - temp_2[1])
                slopes.append(slope)
        slopecounter = 0
        for t in range(len(slopes)):
            if abs(slopes[t]) <= 1:
                slopecounter +=1
        if slopecounter >= 1:
            removelist.append(i)
        print(slopes)
    
    temp_bucket2 = []
    counted_indices = []
    for l in range(len(temp_bucket)):
        if l in removelist:
            pass
        else:
            temp_bucket2.append(temp_bucket[l])
            counted_indices.append(l)
    
    temp_bucket = temp_bucket2
    
    frame_indices3 = []
    for i in range(len(counted_indices)):
        frame_indices3.append(frame_indices2[counted_indices[i]])
   
    " plot the counted vessels "
    print("Plots of analyzed microbubble firings")
    for i in range(len(frame_indices3)):
        bubbleplot(frame_indices3[i][0],subset,centroid_bucket)
        

                
    #     meanslopes = np.mean(slopes)
    #     removeit = False
    #     for k in range(len(slopes)):
    #         if slopes[k] >= 0.2 :
    #             pass
    #         else:
    #             removeit = True
        
    #     if removeit == True:
    #         removelist.append(i)
        
    #     current_temp_bucket = temp_bucket[i]
    #     m,b = polyfit(current_temp_bucket[0], current_temp_bucket[1], 1) 
    
    
    # temp_bucket2 = []
    # counted_indices = []
    # for l in range(len(temp_bucket)):
    #     if l in removelist:
    #         pass
    #     else:
    #         temp_bucket2.append(temp_bucket[l])
    #         counted_indices.append(l)
       
    # temp_bucket = temp_bucket2
        
    
                
    # for i in range(len(counted_indices)):
    #     bubbleplot(counted_indices[i],subset,centroid_bucket)
    
    
    distance_bucket = []   # holds the distances between each of the centroids 
    
    for j in range(len(temp_bucket)):
      temp_bucket[j] = np.flip(temp_bucket[j])   # just convention 
      inner_distance_bucket = []
      for i in range(len(temp_bucket[j])-1):
          x_1 = temp_bucket[j][i][0] * (0.0189)   # these factors multiplied are scaling constants to convert pixels to mm
          x_2 = temp_bucket[j][i+1][0] * (0.0189)
          y_1 = temp_bucket[j][i][1] * (0.0174)
          y_2 = temp_bucket[j][i+1][1] * (0.0174)
          distance = sqrt( (x_1-x_2)**2 + (y_1-y_2)**2)
          inner_distance_bucket.append(distance)
      distance_bucket.append(inner_distance_bucket)
            
    # Generate velocities using the parameters of the system, and distances vs frames
        
    
    velocitybin = []
    delta_frame = 1
    for i in range(len(distance_bucket)):
        inner_velocitybin = []
        for j in range(len(distance_bucket[i])):
                inner_velocitybin.append((50*distance_bucket[i][j])  / delta_frame )      # recriprocal of frame rate 50fps (if fps changed, change this line)
        velocitybin.append(inner_velocitybin)
        
    
    
    # generating coloured velocity vectors between centroid coordinates. 
    
    
    import matplotlib.pyplot as plt
    im = bubbleplot(12,subset,centroid_bucket)
    plt.show()
    
    
    x = []
    y = []
    for j in range(len(temp_bucket)):
        plt.figure()
        implot = plt.imshow(im,cmap = 'binary')
        x_inner = []
        y_inner = []
        for i in range(len(temp_bucket[j])):
                       y_inner.append(temp_bucket[j][i][1])
                       x_inner.append(temp_bucket[j][i][0])
        x.append(x_inner)
        y.append(y_inner)
        d = colorline(x_inner,y_inner,velocitybin[j],norm = plt.Normalize(0,10),linewidth = 4,cmap = 'Blues')
        plt.colorbar()
        plt.colorbar(d)
        
    if y == []:
        print("NO CENTROIDS DETECTED")
        # Polygon_bucket.remove(Polygon_bucket[vesselnumber])
        # Rectangle_bucket.remove(Rectangle_bucket[vesselnumber])
        # attemptvessel -= 1
        # vesselnumber -= 1
        X_bucket.append([])
        Y_bucket.append([])
        v_bucket.append([])
        variance_bucket.append([]) 
        vesseltypebucket.append(vesseltype)
        intensity_threshold_bucket.append(intensity_threshold)
                
        if singlevesselmode == True:
            # Polygon_bucket.remove(Polygon_bucket[vesselnumber])
            # Rectangle_bucket.remove(Rectangle_bucket[vesselnumber])
            pass
        else:
            pass
            
    # averaging velocity colour plots to segment the vessel 
    
    # 1. generating the segmented regions of the vessel 
            
    if y != []: 
        
        plt.imshow(subset[:,:,12])
        for i in range(len(mypolygon)):
            highlight_cell(mypolygon[i][0],mypolygon[i][1], color="red", linewidth=2)
        
        box_y = y[0]
        for i in range(len(y)):
            if len(y[i]) >= len(box_y):
                box_y = y[i]
                box_x = x[i]
            else:
                pass
                
        segymax = max(box_y)+2
        segymin = min(box_y)-2
        segxmax = max(box_x)+2
        segxmin = min(box_x)-2
        
        num_of_segments = len(box_y) 
        
        y_step = (segymax-segymin )/ num_of_segments
        x_step = (segxmax-segxmin) / num_of_segments
        
        # generate num_of_segments amount of rectangles to segment vessel 
        
        seg_rectangles = []
        avg_velocities = []
        for i in range(num_of_segments):
            lowleftcorner = (segxmin ,segymax - i*y_step)
            lowrightcorner = (segxmax, segymax - i*y_step)
            upleftcorner =  (segxmin,segymax - (i+1)*y_step)
            uprightcorner = (segxmax,segymax - (i+1)*y_step)
            rectanglepoints = [lowleftcorner,lowrightcorner,uprightcorner,upleftcorner]
            seg_rectangles.append(Polygon(rectanglepoints))
            
        avg_velocity_bucket = []
        for i in range(num_of_segments):
            avg_velocity_bucket.append([])
        
        
        # populate avg_velocities array with corresponding velocities 
        for j in range(len(velocitybin)):    # iterate over spike trains
          for k in range(len(velocitybin[j])): # iterate over individual velocities in a given train 
            for i in range(len(seg_rectangles)): # iterate over seg_rectangles
                
                point1 = Point((x[j][k],y[j][k]))
                point2 = Point((x[j][k+1],y[j][k+1]))
                if seg_rectangles[i].contains(point1):
                    avg_velocity_bucket[i].append(velocitybin[j][k])
                else:
                    pass
                if seg_rectangles[i].contains(point2):
                    avg_velocity_bucket[i].append(velocitybin[j][k])
                else:
                    pass
                
                test_point = seg_rectangles[i].representative_point()
                test_point_coords = test_point.coords[:]
                if y[j][k+1] <= test_point_coords[0][1] <= y[j][k]:
                    avg_velocity_bucket[i].append(velocitybin[j][k])
                else:
                    pass
    
        
        inner_variance_bucket = []
        for i in range(len(seg_rectangles)):
            square_var = []
            for j in range(len(avg_velocity_bucket[i])):
                    square_var.append(avg_velocity_bucket[i][j]**2)
            variance = np.mean(square_var) - (np.mean(avg_velocity_bucket[i])**2)
            inner_variance_bucket.append(variance)
           
        
        
        " Compute averaged segment velocity over the 1000 frames " 
        
        for i in range(len(avg_velocity_bucket)):
            avg_velocity_bucket[i] = np.mean(avg_velocity_bucket[i])
        
        
        
        if vesseltype == "V":
            mapcolor = "Blues"
        elif vesseltype == "A":
            mapcolor = "Reds"
        else:
            print("Error: Vessel input only takes two options: V or A ")
        
        print("Frame 969")
        im = bubbleplot(969,subset,centroid_bucket) 
        implot = plt.imshow(im,cmap = 'copper')
        d = colorline(box_x,box_y,avg_velocity_bucket,norm = plt.Normalize(0,10),linewidth = 4,cmap = mapcolor)
        plt.colorbar(label = "Intensity (Power/u_area^2)")
        plt.colorbar(d,label = "Microbubble Velocity (mm/s" )
        plt.show()
        
        
        
        plt.figure(figsize = (24,20))
        im = MIP
        implot = plt.imshow(im,cmap = 'binary')
        box_x_MIP = [x + xtransform_min for x in box_x]
        box_y_MIP = [y + ytransform_min for y in box_y]
        d = colorline(box_x_MIP,box_y_MIP,avg_velocity_bucket,norm = plt.Normalize(0,10),linewidth = 8,cmap = mapcolor)
        plt.title("Vessel Velocities on Rat Saggital Slice",fontsize = 50)
        cbar1 = plt.colorbar(implot,label = "Intensity (Power/u_area^2)",fraction = 0.035,pad = 0.1)
        cbar2 = plt.colorbar(d,label = "Microbubble Velocity (mm/s)",fraction = 0.035,pad = 0.08)
        cbar1.set_label(label="Intensity (Power/u_area^2)", size=30)
        cbar2.set_label(label="Microbubble Velocity (mm/s)", size=30)
        cbar1.ax.tick_params(labelsize = 30)
        cbar2.ax.tick_params(labelsize = 30)
        plt.xticks(fontsize = 30)
        plt.yticks(fontsize = 30)
        plt.savefig("velocities.png")
        plt.show()
        
        
        t1 = time.time()
        
        print("Time elapsed:", (t1-t0 )/ 60 , "minutes")
        # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        
        if attemptvessel == 1:

            uu = input("Happy with the vessel? (Y or N):")
            if uu == "Y":
                X_bucket.append(box_x_MIP)
                Y_bucket.append(box_y_MIP)
                v_bucket.append(avg_velocity_bucket)
                variance_bucket.append(inner_variance_bucket)
                vesseltypebucket.append(vesseltype)
                intensity_threshold_bucket.append(intensity_threshold)
            else:
                vesselnumber -= 1
                Polygon_bucket.pop()
                Rectangle_bucket.pop()
                   
            
            uuu = input("Do another vessel? (input Y or N) :")
            if uuu == "Y":
                datacollectionperiod = "not done"
                # attemptvessel += 1
                singlevesselmode = True
            elif uuu == "N":
                datacollectionperiod = "done"
            else:
                pass
        
        
        
        else:
                X_bucket.append(box_x_MIP)
                Y_bucket.append(box_y_MIP)
                v_bucket.append(avg_velocity_bucket)
                vesseltypebucket.append(vesseltype)
                intensity_threshold_bucket.append(intensity_threshold)
                variance_bucket.append(inner_variance_bucket) 
                
        
    else:
        pass
        
    if y != []:   
    
       fig = plt.figure(figsize = (24,20))
       im = MIP
       implot = plt.imshow(im,cmap = 'binary')
       b = []
       r = []
       for i in range(vesselnumber+1): 
            if vesseltypebucket == []:
                pass
            elif vesseltypebucket[i] == "V":
                b_inner = colorline(X_bucket[i],Y_bucket[i],v_bucket[i],norm = plt.Normalize(0,10),linewidth = 8,cmap = "Blues")
                b.append(b_inner)
            elif vesseltypebucket[i] == "A":
                r_inner = colorline(X_bucket[i],Y_bucket[i],v_bucket[i],norm = plt.Normalize(0,10),linewidth = 8,cmap = "Reds")
                r.append(r_inner)  
            else:
                pass
           
              
       plt.title("Vessel Velocities on Rat Saggital Slice",fontsize = 50)
       cbar1 = plt.colorbar(implot,label = "Intensity (Power/u_area^2)",fraction = 0.035,pad = 0.1)
       cbar1.set_label(label="Intensity (Power/u_area^2)", size=30)
       cbar1.ax.tick_params(labelsize = 30)
       
       if b != []:
           cbar2 = plt.colorbar(b[0],label = "Microbubble Velocity (mm/s)",fraction = 0.035,pad = 0.08)
           cbar2.set_label(label="Venular Microbubble Avg Velocity (mm/s)", size=20)
           cbar2.ax.tick_params(labelsize = 30)
       else:
           pass
       if r != []:
           cbar3 = plt.colorbar(r[0],label = "Microbubble Velocity (mm/s)",fraction = 0.04,pad = 0.08)
           cbar3.set_label(label="Arteriolar Microbubble Avg Velocity (mm/s)", size=20)
           cbar3.ax.tick_params(labelsize = 30)
       else:
           pass

       plt.xticks(fontsize = 30)
       plt.yticks(fontsize = 30)
       plt.savefig("velocities.png")
       plt.show()
           
       








       
       

"Save data "

numpy.save("X" + str(vesselnumber),X_bucket)
numpy.save("Y" + str(vesselnumber),Y_bucket)
numpy.save("v" + str(vesselnumber),v_bucket)
numpy.save("vesseltypes" + str(vesselnumber), vesseltypebucket)
numpy.save("averageVbin" + str(vesselnumber), avg_velocity_bucket)
numpy.save("variance" + str(vesselnumber), variance_bucket)
numpy.save("intensity_thresh" + str(vesselnumber), intensity_threshold_bucket) 
       
        
       
       
"""      
       
Average arteriolar velocity 

Average venular velocity 

"""
       
art_vel = []
ven_vel = []

for i in range(len(vesseltypebucket)):
    if vesseltypebucket[i] == "A":
        art_vel.append(v_bucket[i])
    elif vesseltypebucket[i] == "V":
        ven_vel.append(v_bucket[i])
    else:
        pass
    
art_vel1 = []
ven_vel1 = []
for i in range(len(art_vel)):
    art_vel1 = art_vel1 + art_vel[i]
for j in range(len(ven_vel)):
    ven_vel1 = ven_vel1 + ven_vel[j]

avg_a_v = np.mean(art_vel1)
avg_v_v = np.mean(ven_vel1)

print("Average Arteriolar Velocity =", round(avg_a_v,4), "mm/s" )
print("Average Venular Velocity =", round(avg_v_v,4), "mm/s")

