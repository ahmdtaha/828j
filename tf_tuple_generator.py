from easydict import EasyDict as edict
import argparse
import constants as const
import cv2
import imageio
import numpy as np
import os.path as osp
import pickle
import time
import traceback
import tensorflow as tf
import tuple_generator_utils as gen_utils
import sys


global class_names_all


def _decode(filename_tf, cfg):
    input_video_path = tf.string_join([[cfg.data_path + '/'], filename_tf])
    input_video_path = tf.reshape(input_video_path, [-1])
    center_frames, motion_enc, class_labels, filenames_tiled = tf.py_func(
        _split_into_train_tuples, [input_video_path[0]], [
        tf.float32, tf.float32, tf.int32, tf.string])

    return center_frames, motion_enc, class_labels, filenames_tiled


def _split_into_train_tuples(video_path, num_frames=6, step=15,
                             sampling_step=2):
    """
    Splits a video into consecutive but non-overlapping sequences.

    Args:
        video_path (str): path to source video (e.g. *.avi).
        num_frames (int): size of each sequence.
        step (int): "Default" step between frames within a sampled sequence. It
            can be readjusted for short videos.

    Returns:
        center_framess: list of center frames for each sampled sequence. Each
            frame is a 3D array (height x widht x 3).
        stacks_of_diffs: list of stack_of_diffs for each sampled sequence. Each
            stack is a 3D array (height x width x num_frames-1).
    """
    global class_names_all

    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except:
        traceback.print_exc()
        sys.stderr.write('Error: failed to read video %s\n' % video_path)
        print('Error: failed to read video %s\n' % video_path)

    total_num_frames = video._meta['nframes']
    # FIXME: if you want to exclude short videos, remove them as preprocessing!
    # if total_num_frames <= 60:
    #     sys.stderr.write('Video %s is too small\n' % video_path)
    #     print('Video %s is too small\n' % video_path)
    #     return [], [], [], []
    if total_num_frames <= 60:
        step = total_num_frames // (num_frames - 1)
    elif total_num_frames <= 75:  # then re-adjust step
        if total_num_frames >= 71:
            step = 14
        elif total_num_frames >= 66:
            step = 13
        else:
            step = 12

    chunks = []
    for start_frame in range(0, total_num_frames, sampling_step):
        chunk = np.arange(
            start_frame, start_frame + (num_frames - 1) * step + 1, step)
        if chunk[-1] < total_num_frames:
            chunks.append(chunk)
            start_frame = chunk[-1] + 1
        else:
            # # if remaining frames cannot complete a chunk, then center
            # # previous chunks so that leftovers are equally two-sided
            # shift = (total_num_frames - start_frame + 1) // 2
            # for i in range(len(chunks)):
            #     chunks[i] = chunks[i] + shift
            break

    center_frames_indices = map(lambda x: (x[-1] + x[0]) // 2, chunks)
    center_frames, stacks_of_diffs = gen_utils.split_into_tuples(
        video_path, chunks, augment_tuples=True)

    center_frames = np.float32(center_frames)
    stacks_of_diffs = np.float32(stacks_of_diffs)
    # FIXME: next line assumes unix paths (won't work for windows). Use regex
    # to split on both / and \
    class_name = video_path.split('/')[-2]
    class_label = np.int32(class_names_all.index(class_name))
    path_no_ext, _ = osp.splitext(video_path)
    filenames = ['%s-frame_%d' % (path_no_ext, x) for x in
                 center_frames_indices]

    return center_frames, stacks_of_diffs, [class_label] * len(chunks), \
        filenames


def _shufle_motion():
    return None


def _build_supervised_input_for_train(filenames_tf, cfg, num_threads):
    center_frames, motion_encodings, class_labels, filenames_tf_tiled = _decode(
        filenames_tf, cfg)
    # TODO: make sure to cast read data to the correct dtype

    # Data augmentation
    if cfg.augmentation_flag:
        # center_frame, motion_encoding = augment_data(center_frame,
        #                                              motion_encoding)
        # raise NotImplementedError  # TODO
        None

    # TODO: check if values for the capacity and min_after_dequeue are not bad
    example_queue = tf.RandomShuffleQueue(
        capacity=8*cfg.batch_size, min_after_dequeue=6*cfg.batch_size,
        dtypes=[tf.float32, tf.float32, tf.int32, tf.string],
        shapes=[[cfg.img_height, cfg.img_width, cfg.img_channels],
                [cfg.img_height, cfg.img_width, cfg.context_channels], [], []])

    example_enqueue_op = example_queue.enqueue_many(
        [center_frames, motion_encodings, class_labels, filenames_tf_tiled])

    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read batch
    center_frame, motion_encoding, class_label, tuple_name = \
        example_queue.dequeue_many(cfg.batch_size)

    return center_frame, motion_encoding, class_label, tuple_name


def _build_supervised_input_for_test():
    return None, None, None, None


def _build_unsupervised_input_for_train():
    return None, None, None, None


def _build_unsupervised_input_for_test():
    return None, None, None, None


def build_input(data_path, input_list_filepath, activities_list_path,
                batch_size, supervision_mode, run_mode, log_root,
                num_threads=16):

    global class_names_all

    cfg = edict()
    cfg.data_path = data_path
    cfg.batch_size = batch_size
    cfg.supervision_mode = supervision_mode
    cfg.run_mode = run_mode
    cfg.log_root = log_root
    cfg.augmentation_flag = True if run_mode == 'train' else False
    cfg.img_height = const.frame_height
    cfg.img_width = const.frame_width
    cfg.img_channels = const.frame_channels
    cfg.context_channels = const.context_channels

    with open(activities_list_path, 'r') as f:
        class_names_all = [x.strip() for x in f.readlines()]

    filename_queue = tf.train.string_input_producer([input_list_filepath])
    reader = tf.TextLineReader()
    _, filenames_tf = reader.read(filename_queue)

    if supervision_mode == 'supervised':
        if run_mode == 'train' or run_mode == 'eval':
            center_frame, motion_encoding, class_label, tuple_name = \
               _build_supervised_input_for_train(filenames_tf, cfg, num_threads)
        elif run_mode == 'test':
            # center_frame, motion_encoding, tuple_name, class_label = \
            #     _build_supervised_input_for_test()
            raise NotImplementedError  # TODO
        else:
            raise ValueError('Unknow run_mode: %s' % run_mode)
    elif supervision_mode == 'unsupervised':
        if run_mode == 'train' or run_mode == 'eval':
            # center_frame, motion_encoding, tuple_name, class_label = \
            #     _build_unsupervised_input_for_train()
            raise NotImplementedError  # TODO
        elif run_mode == 'test':
            # center_frame, motion_encoding, tuple_name, class_label = \
            #     _build_unsupervised_input_for_test()
            raise NotImplementedError  # TODO
        else:
            raise ValueError('Unknow run_mode: %s' % run_mode)
    else:
        raise ValueError('Unknow supervision_mode: %s' % supervision_mode)

    return center_frame, motion_encoding, class_label, tuple_name
