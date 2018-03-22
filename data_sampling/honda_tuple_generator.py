import sys
sys.path.append('../')

import os
import glob
import numpy as np
import pickle as pkl
import tensorflow as tf
import random
import pdb
import imageio
import configuration as config

def tuple_generator(supervised, sessions, batch_size=8, num_threads=2, shuffled=True, stride=11,
        frame_dir=config.honda_session_path+'/frames/',
        label_dir=config.honda_session_path+'/labels/'):
    """
    Generator of tuples

    If supervised == False, it should return a tuple of (center RGB frame, stack of six RGB frames around the center frame)
    if supervised == True, it should return a tuple of (center RGB frame, stack of six RGB frames around the center frame, Event_label)

    For 30fps, The time difference between consecutive frames = 15 
    For example, stack of six frames time = [ 29  44  59  74  89 104], center frame = 67 = (104 + 29) // 2.

    Arguments:
    supervised -- bool
    sessions -- list of str, containing session ids
    batch_size -- int, 
    num_threads -- int, the number of threads used for preprocessing
    shuffled -- bool
    stride -- int, moving stride for center image, used for unsupervised mode
    """

    # collect image paths
    all_paths = {}
    for session in sessions:
        frame_paths = glob.glob(frame_dir+session+'/frame_*.jpg')
        idx = [int(path.split('_')[-1].split('.')[0]) for path in frame_paths]
        frame_paths_sorted = [path for _,path in sorted(zip(idx, frame_paths))]
        
        all_paths[session] = frame_paths_sorted

    paths = []
    label = []
    if supervised:
        for key in all_paths:
            N = len(all_paths[key])
            #print(type(key))
            #print(type(label_dir))
            # load label
            annotation = pkl.load(open(label_dir+key+'_goal.pkl', 'rb'),fix_imports=True)    # goal-oriented events
            for i in range(len(annotation['G'])):
                end = annotation['s'][i+1]
                start = annotation['s'][i]
                
                if end - start > 5:    # ignore short (background) clips
                    center_idx = (start + end) // 2

                    if center_idx < 5 or center_idx > N-6:
                        continue    # ignore the invalid sample
                    
                    center_path = all_paths[key][center_idx]
                    p1 = all_paths[key][center_idx-5]
                    p2 = all_paths[key][center_idx-3]
                    p3 = all_paths[key][center_idx-1]
                    p4 = all_paths[key][center_idx+1]
                    p5 = all_paths[key][center_idx+3]
                    p6 = all_paths[key][center_idx+5]

                    paths.append((center_path, p1,p2,p3,p4,p5,p6))
                    label.append(annotation['G'][i])
                
    else:
        for key in all_paths:
            N = len(all_paths[key])
            for center_idx in range(stride/2+1, N-stride/2-1, stride):
                center_path = all_paths[key][center_idx]
                p1 = all_paths[key][center_idx-5]
                p2 = all_paths[key][center_idx-3]
                p3 = all_paths[key][center_idx-1]
                p4 = all_paths[key][center_idx+1]
                p5 = all_paths[key][center_idx+3]
                p6 = all_paths[key][center_idx+5]
                paths.append((center_path, p1,p2,p3,p4,p5,p6))
                label.append(center_idx)    # fake label for convenient

    # uncomment here to check whether paths are correct
    #pdb.set_trace()

    dataset = tf.data.Dataset.from_tensor_slices((paths, label))

    def _input_parser(path, lab):
        center_image = tf.image.decode_image(tf.read_file(path[0]))
        I1 = tf.expand_dims(tf.image.decode_image(tf.read_file(path[1])),axis=0)
        I2 = tf.expand_dims(tf.image.decode_image(tf.read_file(path[2])),axis=0)
        I3 = tf.expand_dims(tf.image.decode_image(tf.read_file(path[3])),axis=0)
        I4 = tf.expand_dims(tf.image.decode_image(tf.read_file(path[4])),axis=0)
        I5 = tf.expand_dims(tf.image.decode_image(tf.read_file(path[5])),axis=0)
        I6 = tf.expand_dims(tf.image.decode_image(tf.read_file(path[6])),axis=0)

        return center_image, tf.concat([I1,I2,I3,I4,I5,I6], axis=0), lab

    dataset = dataset.map(_input_parser, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(4*num_threads*batch_size)

    if shuffled:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)


    return dataset


if __name__ == '__main__':

#    train_sessions = ['201706071021', '201704111138', '201703061700']    # for example
    with open(config.honda_session_path+'/train_session.txt','r') as fin:
        train_sessions = fin.read().strip().split('\n')
    with open(config.honda_session_path+'/val_session.txt','r') as fin:
        val_sessions = fin.read().strip().split('\n')

    train_data = tuple_generator(supervised=True, sessions=train_sessions)
#    train_data = tuple_generator(supervised=False, sessions=train_sessions)
    print(train_data.output_types)
    train_iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                    train_data.output_shapes)
    next_train = train_iterator.get_next()
    train_init_op = train_iterator.make_initializer(train_data)

    val_data = tuple_generator(supervised=True, sessions=val_sessions, shuffled=False)
#    val_data = tuple_generator(supervised=False, sessions=val_sessions, shuffled=False)
    val_iterator = tf.data.Iterator.from_structure(val_data.output_types,
                                                    val_data.output_shapes)
    next_val = val_iterator.get_next()
    val_init_op = val_iterator.make_initializer(val_data)
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(3):

            sess.run(train_init_op)
            while True:
                try:
                    imgs, neighbors, labels = sess.run(next_train)
                    print(labels)

                except tf.errors.OutOfRangeError:
                    print("Epoch %d done!" % (epoch+1))
                    break

            sess.run(val_init_op)
            while True:
                try:
                    imgs, neighbors, labels = sess.run(next_val)
                    print(labels)

                except tf.errors.OutOfRangeError:
                    break
