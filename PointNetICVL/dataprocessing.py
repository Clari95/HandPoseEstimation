# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf
import numpy as np

pointSize = 256
def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

def preprocessPoint(points,joints):
    boundingBoxSize = 0.15
    validIndicies = np.logical_and(np.logical_and(np.abs(points[:,0])<boundingBoxSize, np.abs(points[:,1])<boundingBoxSize), np.abs(points[:,2])<boundingBoxSize)
    points = points[validIndicies, :]
    if len(points)==0:
        points = np.zeros((pointSize*2,3), dtype = np.float32)
    while len(points) < pointSize*2:
        points = np.repeat(points, 2, axis = 0)

    randInidices =np.arange(len(points))
    np.random.shuffle(randInidices)
    points_sampled = points[randInidices[0:pointSize,],:]
    return points_sampled, joints



def preprocessPoint_augment(points,joints):
    boundingBoxSize = 0.15
    joints = joints.reshape(16,3)

    #random scaling
    randScale = np.maximum(np.minimum(np.random.normal(1.0, 0.05),1.25),0.75)
    points = points*randScale
    joints = joints*randScale
    boundingBoxSize =boundingBoxSize*randScale

    #random rotation around camera view axis (z-axis)
    randAngle = -math.pi+2*math.pi*np.random.rand(1)
    (points[:,0],points[:,1]) = rotate((0,0), (points[:,0],points[:,1]), randAngle)
    (joints[:,0],joints[:,1]) = rotate((0,0), (joints[:,0],joints[:,1]), randAngle)

    #random translation
    randTrans = np.float32(np.maximum(np.minimum(np.random.normal(0.0, 7.0, (3,)),25.0),-25.0)/1000)
    randTrans[2] = np.float32(np.maximum(np.minimum(np.random.normal(0.0, 9.0, (1,)),27.0),-27.0)/1000)
    joints = joints + randTrans
    points = points + randTrans

    validIndicies = np.logical_and(np.logical_and(np.abs(points[:,0])<boundingBoxSize, np.abs(points[:,1])<boundingBoxSize),
                                   np.abs(points[:,2])<boundingBoxSize)

    points = points[validIndicies, :]

    if len(points)==0:
        points = np.zeros((2048,3), dtype = np.float32)
    while len(points) < 1024:
        points = np.repeat(points, 2, axis = 0)


    randInidices = np.arange(len(points))
    np.random.shuffle(randInidices)
    points_sampled = points[randInidices[0:pointSize,],:]

    # rotation around x axis
    randAngle = -math.pi+2*math.pi*np.random.rand(1)  #[-180 180] degrees
    (points_sampled[:,1],points_sampled[:,2]) = rotate((0,0), (points_sampled[:,1],points_sampled[:,2]), randAngle)
    (joints[:,1],joints[:,2]) = rotate((0,0), (joints[:,1],joints[:,2]), randAngle)

    # rotation around y axis
    randAngle = (-math.pi+2*math.pi*np.random.rand(1)) * 0.01  #[-5.4, +5.4] degrees
    (points_sampled[:,0],points_sampled[:,2]) = rotate((0,0), (points_sampled[:,0],points_sampled[:,2]), randAngle)
    (joints[:,0],joints[:,2]) = rotate((0,0), (joints[:,0],joints[:,2]), randAngle)

    joints = joints.reshape(48)

    points_sampled[np.isnan(points_sampled)] = 0.0
    joints[np.isnan(joints)] = 0.0

    return points_sampled, joints



