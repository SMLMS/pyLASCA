#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:38:21 2020

@author: malkusch
"""

import matplotlib.pyplot as plt
from shapely.geometry import Polygon
# =============================================================================
# p1 = Polygon([(0,0), (1.1,1), (1.1,0)])
# p2 = Polygon([(0,0.8), (1,-0.1), (1,0.8)])
# =============================================================================
p1 = Polygon([(0,0),
              (0,1),
              (1,1),
              (1,0)])
p2 = Polygon([(-0.25,0.25),
              (-0.25, 0.75),
              (0.25, 0.75),
              (0.25, 0.25)])
print(p1.intersects(p2))
p3 = p1.intersection(p2)

print(p1.area)
print(p2.area)
print(p3.area)

c1 = p1.centroid
c2 = p2.centroid
c3 = p3.centroid

print(c1.distance(c2))
print(c1.distance(c3))
print(c2.distance(c3))


x1,y1 = p1.exterior.xy
x2,y2 = p2.exterior.xy
x3,y3 = p3.exterior.xy

plt.plot(x1, y1, color='#6699cc', alpha=0.7,
    linewidth=3, solid_capstyle='round', zorder=2)
plt.plot(x2, y2, color='#6699cc', alpha=0.7,
    linewidth=3, solid_capstyle='round', zorder=2)
plt.plot(x3, y3, color='red', alpha=0.7,
    linewidth=3, solid_capstyle='round', zorder=2)

plt.show()