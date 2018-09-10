# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     euclidean_distance_plot
   Description :  绘制3维空间的Euclidean 距离
   Author :       Stephen
   date：          2018/9/3
-------------------------------------------------
   Change Activity:
                   2018/9/3:
-------------------------------------------------
"""
__author__ = 'Stephen'

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

coords1 = [1, 2, 3]
coords2 = [4, 5, 6]

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter((coords1[0], coords2[0]),
           (coords1[1], coords2[1]),
           (coords1[2], coords2[2]),
           color="y", s=150)

ax.plot((coords1[0], coords2[0]),
        (coords1[1], coords2[1]),
        (coords1[2], coords2[2]),
        color="r")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.text(x=2.5, y=3.5, z=4.0, s='d = 5.19')

plt.title('Euclidean distance between 2 3D-coordinates')

plt.show()
