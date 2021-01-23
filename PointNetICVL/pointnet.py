"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation ResNet variant implemented was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

"""

# Python 2 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Imports
import tensorflow as tf
import PointNetLearn as PNL
import numpy as np
import math
import sys
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

pointSize = 256


def PointNet(points, training_var, batch_size, bn_decay=None):

    input_image = tf.expand_dims(points, -1)

    points = tf.reshape(points, [-1,pointSize,3, 1])
    net = tf_util.conv2d(points, 64, [1, 3],  # input_image
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=training_var,
                         scope='conv1', bn_decay=bn_decay)
    # print('netshape')
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=training_var,
                         scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=training_var,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=training_var,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=training_var,
                         scope='conv5', bn_decay=bn_decay)
    print('before maxpool')
    print(net.get_shape())

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [pointSize, 1],
                             padding='VALID', scope='maxpool')
    print('after maxpool')

    print(net.get_shape())


    print ('batch_size', batch_size)
    # MLP on global point cloud vector
    print('nethspe: ', net.get_shape())
    net = tf.reshape(net, [-1, 1024])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=training_var,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=training_var,
                                  scope='fc2', bn_decay=bn_decay)
    pose = tf_util.fully_connected(net, 48, activation_fn=None, scope='fc3')
    print('posehape: ', pose.get_shape())
    return pose
    """PointNet model
    Args:
        points: Input points to be processed  BatchSize x N x 3
    Returns:
        pose3D: 3D joint positions (16x3 = 48)
    """ 


def loss_pointnet(poseGT, pose3D):                  #next_joints,  pose3DPrediction
    poseDiff = pose3D - poseGT
    poseDiff = tf.abs(tf.reshape(poseDiff,[-1,16,3]))
    eucLoss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(poseDiff,2),axis = 2)))
    tf.summary.scalar('eucLoss', eucLoss)

    poseLossX = tf.squeeze(tf.reduce_mean(poseDiff[:,:,0]))
    poseLossY = tf.squeeze(tf.reduce_mean(poseDiff[:,:,1]))
    poseLossZ = tf.squeeze(tf.reduce_mean(poseDiff[:,:,2]))
    tf.summary.scalar('poseLossX', poseLossX)
    tf.summary.scalar('poseLossY', poseLossY)
    tf.summary.scalar('poseLossZ', poseLossZ)

    poseLoss = tf.squeeze(tf.nn.l2_loss(poseDiff))
    tf.summary.scalar('PoseL2Loss', poseLoss)
    return poseLoss, pose3D, eucLoss

def train_net(loss, learning_rate):
    """Sets up the training operations.
    Creates a summarizer to track the loss and the learning rate over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    The operation returned by this function is what must be passed to the sess.run() call to cause the model to train.
    Args:
        loss: Loss tensor
        learning_rate: The learning rate to be used for Adam.
    Returns:
        train_pointnet_op: The Op for training.
    """

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('totalLoss', loss)
    tf.summary.scalar('learning_rate_posenet', learning_rate)

    # Create the Adam optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_pointnet_op = optimizer.minimize(loss, global_step=global_step)
    return train_pointnet_op
