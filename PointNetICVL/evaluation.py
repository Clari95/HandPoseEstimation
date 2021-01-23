# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import time
import tensorflow as tf
import numpy as np

def inference_for_result(sess,eval_op, pose_op, initializable_iterator, handle, data_handle, training_var, steps_per_epoch):
    # Start timer
    start_time = time.time()
    # Initialize iterator
    sess.run(initializable_iterator.initializer)
    # OPEN FILE
    F = open("results.txt", "wb")
    # Iterate over training dataset
    for i in range(steps_per_epoch):
        if (i%100 == 0):
            print(time.time() - start_time)
            print (i)
            start_time = time.time()
        loss, pose_to_save = sess.run([eval_op, pose_op], feed_dict={handle: data_handle, training_var: False})
        pose_to_save = np.squeeze(pose_to_save)
        np.savetxt(F, pose_to_save, delimiter="\t", newline="\n", fmt="%f")
    # CLOSE FILE
    F.close()


def inference_Batch(sess, iterator, eval_op, eucLoss, handle, data_handle, training_var, batchsize):
    start_time = time.time()
    # sess.run(iterator.initializer)
    _,euclideanLoss = sess.run([eval_op,eucLoss], feed_dict={handle: data_handle, training_var: False})
    #print('Type of euclideanLoss')
    #print(euclideanLoss.dtype)


    eval_euclidean_avg = math.sqrt(euclideanLoss/batchsize)

    # Print training data evaluation results
    duration = time.time() - start_time
    print('THIS BATCH: Num examples: %d  Avg pdf euclidean error: %.8f  Time for evaluation: %.2f sec' % (
            batchsize, eval_euclidean_avg, duration))


def inference_test(sess, iterator, eval_op, eucLoss, handle, data_handle, training_var, batchsize, stepsize):
    start_time = time.time()
    sess.run(iterator.initializer)
    euclideanLoss = 0
    for i in range(stepsize):
        _,euclideanLoss = sess.run([eval_op, eucLoss], feed_dict={handle: data_handle, training_var: False})
        euclideanLoss += math.sqrt(euclideanLoss / batchsize)

    eval_loss_avg = math.sqrt(euclideanLoss / (batchsize*stepsize))

    duration = time.time() - start_time
    print('VALIDATION: Num examples: %d  Avg l2 loss: %.8f  Time for evaluation: %.2f sec'  % (
            batchsize*stepsize, eval_loss_avg, duration))
