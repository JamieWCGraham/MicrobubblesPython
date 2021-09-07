#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:12:00 2019

@author: jamiegraham
"""

import numpy as np 
import matplotlib.pyplot as plt
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

def multicolored_lines():

    x = np.linspace(0, 4. * np.pi, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    lc = colorline(x, y, cmap='hsv')
    plt.colorbar(lc)
    plt.xlim(x.min(), x.max())
    plt.ylim(-1.0, 1.0)
    plt.show()

def colorline(x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
         z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# multicolored_lines()