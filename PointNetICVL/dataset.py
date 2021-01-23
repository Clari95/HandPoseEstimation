# Python 2 Compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports 
import math
import tensorflow as tf
from dataprocessing import *


def dataset_filenames(data_dir):
    # Training and validation data locations
    training_filenames = ["/ICVLTrainPointData_1",
                          "/ICVLTrainPointData_2",
                          "/ICVLTrainPointData_3",
                          "/ICVLTrainPointData_4",
                          "/ICVLTrainPointData_5",
                          "/ICVLTrainPointData_6",
                          "/ICVLTrainPointData_7",
                          "/ICVLTrainPointData_8",
                          "/ICVLTrainPointData_9",
                          "/ICVLTrainPointData_10",
                          "/ICVLTrainPointData_11",
                          "/ICVLTrainPointData_12",
                                                ]

    validation_filenames = ["/ICVLTestPointData_seq1_1",
                            "/ICVLTestPointData_seq2_1",
                            ]

    test_filenames = ["/ICVLTestPointData_seq1_1",
                      "/ICVLTestPointData_seq2_1",
                      ]

    # Append directory where data is located
    training_filenames = [data_dir+s for s in training_filenames]
    validation_filenames = [data_dir+s for s in validation_filenames]
    test_filenames = [data_dir+s for s in test_filenames]

    return training_filenames, validation_filenames, test_filenames


def parse_function(example_proto):
    """Parse through current binary batch and extract images and labels"""
    # Parse through features and extract byte string
    parsed_features = tf.parse_single_example(example_proto,features ={
        'pointCloud': tf.FixedLenFeature([],tf.string),
        'joint': tf.FixedLenFeature([],tf.string)
        },name='features')

    # Decode content into correct types
    points_dec = tf.decode_raw(parsed_features['pointCloud'],tf.float32)
    joint_dec = tf.decode_raw(parsed_features['joint'],tf.float32)
    points_dec = tf.reshape(points_dec, [4000,3])
    # preprocess points
    points_dec, joint_dec = tf.py_func(preprocessPoint,[points_dec, joint_dec],[tf.float32, tf.float32])
    return points_dec, joint_dec


def parse_function_augment(example_proto):
    """Parse through current binary batch and extract images and labels"""
    # Parse through features and extract byte string
    parsed_features = tf.parse_single_example(example_proto,features = {
        'pointCloud': tf.FixedLenFeature([],tf.string),
        'joint': tf.FixedLenFeature([],tf.string)
        },name='features')

    # Decode content into correct types
    points_dec = tf.decode_raw(parsed_features['pointCloud'],tf.float32)
    joint_dec = tf.decode_raw(parsed_features['joint'],tf.float32)
    points_dec = tf.reshape(points_dec, [4000,3])
    # preprocess points
    points_dec, joint_dec = tf.py_func(preprocessPoint_augment,[points_dec, joint_dec],[tf.float32, tf.float32])
    return points_dec, joint_dec



def create_datasets(training_filenames, validation_filenames, test_filenames, handle, batch_size, thread_count, buffer_count):
    # Define the training with Dataset API
    #print('filenames_dataset',training_filenames)
    training_dataset = tf.data.TFRecordDataset(training_filenames)
    #print('trainingdataset', training_dataset)
    #print(tf.TensorShape(training_dataset))
    training_dataset = training_dataset.map(parse_function_augment, num_parallel_calls=thread_count)
    training_dataset = training_dataset.shuffle(buffer_size=buffer_count)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.repeat()
   # print(tf.TensorShape(training_dataset))



    # Define the validation dataset for evaluation
    validation_dataset = tf.data.TFRecordDataset(validation_filenames)
    validation_dataset = validation_dataset.map(parse_function, num_parallel_calls=thread_count)
    validation_dataset = validation_dataset.shuffle(buffer_size=buffer_count)
    validation_dataset = validation_dataset.batch(batch_size)
    validation_dataset = validation_dataset.repeat()

    # Define the evaluation on training dataset dataset
    test_dataset = tf.data.TFRecordDataset(test_filenames)
    test_dataset = test_dataset.map(parse_function, num_parallel_calls=thread_count)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.repeat()

    #print('shapetrainingdataset_dataset', training_dataset.output_shapes)
    # Create a feedable iterator to consume data
    iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
    next_points, next_joints = iterator.get_next()
    #print('nextpointsdataset' , next_points)
    # Define the different iterators
    training_iterator = training_dataset.make_one_shot_iterator()
    validation_iterator = validation_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    return next_points, next_joints, training_iterator, validation_iterator, test_iterator
