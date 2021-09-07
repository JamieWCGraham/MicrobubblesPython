#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:57:22 2019

@author: jamiegraham
"""

import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.pyplot
from numpy import sqrt, abs, exp, int, cos, tan, sin, fft
from scipy.constants import c, h, k, pi, sigma, epsilon_0, e 
import random as random
import time
import scipy as scipy
from scipy import integrate
import scipy.io
import scipy.stats as st
from scipy import signal
# import ROI
import roipoly
import cv2



image = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        image[i][j] = random.random()
        
# plt.figure(figsize = (100,10))
# fig = plt.imshow(image)

import numpy as np
import cv2 as cv2
import cv

# ============================================================================

CANVAS_SIZE = (600,800)

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (4, 4, 4)

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self,image):
        # Let's create our working window and set a mouse callback to handle events
        # cv2.namedWindow(self.window_name, flags=cv2.CV_WINDOW_AUTOSIZE)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)       
        # cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.resizeWindow('image', 900, 900)
        cv2.imshow(self.window_name, image)   
        cv2.waitKey(1)
        cv2.cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = image
            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        canvas = image
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        # cv2.destroyAllWindows()
        return canvas

"""
# ============================================================================

image = np.zeros((80,40))
for i in range(80):
    for j in range(40):
        image[i][j] = random.random()

# image = subset[:,:,12]        
        
# image = plt.imshow(image)
        

if __name__ == "__main__":
    pd = PolygonDrawer("Polygon")
    image2 = pd.run(image)
    cv2.imwrite("polygon.png", image2)
    print("Polygon = %s" % pd.points)

"""
