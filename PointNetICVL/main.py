# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import argparse
import os
import sys
import time
import math
from six.moves import xrange 

import tensorflow as tf
from pointnet import *
from evaluation import *
from dataset import *

# Basic model parameters as external flags.
FLAGS = None
training_var = tf.placeholder(tf.bool)

def run_net():
    """Train the ResNet on the hand image data."""
    # Create the iterators and data placeholders for the datasets
    training_filenames, validation_filenames, test_filenames = dataset_filenames(FLAGS.data_dir)
    handle = tf.placeholder(tf.string, shape=[])
    next_points, next_joints, training_iterator, validation_iterator, test_iterator = \
        create_datasets(training_filenames, validation_filenames, test_filenames, handle, FLAGS.batch_size, 8, 6400)
    print('FLAGS.batch_size', FLAGS.batch_size)
    # Build a Graph that computes predictions from input points
    pose3DPrediction = PointNet(next_points, training_var, FLAGS.batch_size)
    eval_op, pose_op, eucLoss = loss_pointnet(next_joints,  pose3DPrediction)

    # Add the operation that calculates and applies the gradients to the graph
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)	# NEEDED FOR BATCH NORMALIZATION
    with tf.control_dependencies(update_ops):
        train_op = train_net(eval_op, FLAGS.learning_rate)

    # Initialize the global variables
    init = tf.global_variables_initializer()
    var_list = tf.global_variables()
    saver = tf.train.Saver(var_list, max_to_keep=10)

    # Create a session for running operations on the Graph.
    sess = tf.Session()

    # number of steps per epoch
    validation_steps_per_epoch = int(math.ceil(FLAGS.validation_size/FLAGS.batch_size))
    test_steps_per_epoch = int(math.ceil(FLAGS.test_size/FLAGS.batch_size))

    # Create the needed handles to feed the handle placeholder
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())


    # Select mode to run network in
    if FLAGS.run_mode=='inference':
        # RESTORE THE MODEL
        checkpoint_file = '/home/student/ProjectClarissa/log/model.ckpt-250000'
        saver.restore(sess, checkpoint_file)
        # START INFERENCE
        print('START INFERENCE')
        inference_test(sess, validation_iterator, eval_op, eucLoss, handle, validation_handle, training_var, FLAGS.batch_size, validation_steps_per_epoch)

    elif FLAGS.run_mode=='inference_result':
        # RESTORE THE MODEL
        checkpoint_file = '/home/student/ProjectClarissa/log/model.ckpt-250000'
        saver.restore(sess, checkpoint_file)

        # START INFERENCE
        print('START COMPUTE TESTING RESULT')
        inference_for_result(sess, eval_op, pose_op, test_iterator, handle, test_handle, training_var, test_steps_per_epoch)
        # sess,eval_op, pose_op, initializable_iterator, handle, data_handle, training_var, steps_per_epoch    after pose_op eucLoss,

    else:

        # Some control variables
        log_frequency_comp = FLAGS.log_frequency - 1
        eval_frequency = int(math.ceil(FLAGS.train_size/FLAGS.batch_size))
        eval_frequency_comp = eval_frequency - 1

        if FLAGS.run_mode=='training':
            # Run the Op to initialize the variables.
            sess.run(init)

        elif FLAGS.run_mode=='cont_training':
            # Restore the model
            sess.run(init)
            checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt-200000')
            saver.restore(sess, checkpoint_file)
            sess.run(test_iterator.initializer)
            sess.run(validation_iterator.initializer)


        # Start the training loop.
        print('START TRAINING')
        start_time = time.time()
        sess.run(test_iterator.initializer)
        sess.run(validation_iterator.initializer)

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        validation_writer = tf.summary.FileWriter(FLAGS.log_dir + '/eval')

        for steps in xrange(FLAGS.start_step ,FLAGS.max_steps):
            if steps % 1000==1 and steps>100:
                inference_Batch(sess, validation_iterator, eval_op, eucLoss, handle, validation_handle,training_var, FLAGS.batch_size)
            # TRAINING
            if steps % FLAGS.log_frequency == log_frequency_comp:  # Logging for Tensorboard
                # Instantiate SummaryWriters to output summaries and the Graph.

                # Build the summary Tensor based on the TF collection of Summaries.
                merged = tf.summary.merge_all()

                summary, _ = sess.run([merged, train_op], feed_dict={handle: training_handle,  training_var: True})
                train_writer.add_summary(summary,steps)
                train_writer.flush()
                summary, _ = sess.run([merged, eval_op], feed_dict={handle: validation_handle, training_var: False})
                validation_writer.add_summary(summary,steps)
                validation_writer.flush()
            else:
                sess.run(train_op, feed_dict={handle: training_handle,  training_var: True})

            # EVALUATION OF TRAINING
            if steps % eval_frequency == eval_frequency_comp:
                # END OF EPOCH
                duration = time.time() - start_time
                print('Step %d completed. Duration of last %d steps: %.2f sec' % (steps+1, eval_frequency, duration))
                # EVALUATION RUN
                inference_test(sess, validation_iterator, eval_op, eucLoss, handle, validation_handle, training_var,
                               FLAGS.batch_size, validation_steps_per_epoch)

                # RESET TIMER
                start_time = time.time()

            if steps % 10000 == 0:                #10000
                # SAVE A MODEL CHECKPOINT
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=steps)




def main(_):
    if FLAGS.run_mode=='training':
        tf.gfile.MakeDirs(FLAGS.log_dir)
    print('in main...')
    run_net()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--start_step',
      type=int,
      default=200001,
      help='Training step to start with.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=500000,

      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/home/student/ProjectClarissa/dataset/ICVL',
      help='Directory where the dataset lies.'
  )
  parser.add_argument(
      '--train_size',
      type=int,
      default=330000,
      help='Training data size (number of images).'
  )
  parser.add_argument(
      '--validation_size',
      type=int,
      default=1600,
      help='Validation data size (number of images).'
  )
  parser.add_argument(
      '--test_size',
      type=int,
      default=1600,
      help='Validation data size (number of images).'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/home/student/ProjectClarissa/log',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--load_dir',
      type=str,
      default='/home/student/ProjectClarissa/load',
      help='Directory to put the load data.'
  )
  parser.add_argument(
      '--log_frequency',
      type=int,
      default=200,
      help='Frequency (steps) with which data is logged'
  )
  parser.add_argument(
      '--run_mode',
      type=str,
      default='inference_result',
      help='Mode in which network is run: training, cont_training, inference_result'
  )

  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
