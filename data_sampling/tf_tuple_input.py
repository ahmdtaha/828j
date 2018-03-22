import sys
sys.path.append('../')

import matplotlib
matplotlib.use('agg')

from easydict import EasyDict as edict
import argparse
import constants as const
import cv2
import imageio
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import os.path as osp
import pickle
import time
import traceback
import tensorflow as tf


def _decode(filename_tf, cfg):
    input_video_path = tf.string_join([[cfg.data_path + '/'], filename_tf])
    input_video_path = tf.reshape(input_video_path, [-1])
    center_frames, motion_enc, class_labels, filenames_tiled = tf.py_func(
        _split_into_train_tuples, [input_video_path[0]], [
        tf.float32, tf.float32, tf.int32, tf.string])

    return center_frames, motion_enc, class_labels, filenames_tiled


def _get_standard_frame(video, frame_index, crop_flag=False):
    """
    Retreives a specific frame from a video, and rescales it into a pre-defined
    height and width (specified by: const.frame_height and const.frame_width).

    Args:
        video: loaded video with imageio ffmpeg.
        frame_index (int): index of the required frame.

    Returns:
        array (height x width x 3) representing the extracted frame after being
        rescaled to a standard size.
    """
    frame = video.get_data(frame_index)
    if crop_flag:
        None  # TODO: random crop (PROBLEM: static graph issues)
    elif(frame.shape[0] != const.frame_height or
         frame.shape[1] != const.frame_width):
        frame = cv2.resize(frame, (const.frame_width, const.frame_height))

    return frame


def _create_stack_of_diffs(video, frame_indices):
    """
    Creates a stack of differences by converting the specified frames into
    grayscale and taking their differences.

    Args:
        video: loaded video with imageio ffmpeg.
        frame_inicies (list of ints): frame indices the constitute the stack.

    Returns:
        array (stack_size x height x width) representing the stack of diffs.
        stack_size is len(frame_indices) - 1
    """
    num_frames = len(frame_indices)
    stack_of_diffs = np.zeros((const.frame_height, const.frame_width,
                               num_frames - 1))

    ## Temporary change the stack of diff to be  current - next
    for i in range(num_frames - 1):
        current_frame = _get_standard_frame(video, frame_indices[i])
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

        next_frame = _get_standard_frame(video, frame_indices[i+1])
        next_frame= cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        stack_of_diffs[:, :, i] = current_frame.astype(np.int32) - next_frame.astype(np.int32);


    # prev_frame = _get_standard_frame(video, frame_indices[0])
    # # dbg_rgb_path = str(frame_indices[0]) + '_rgb.png'
    # # cv2.imwrite(dbg_rgb_path, cv2.cvtColor(prev_frame, cv2.COLOR_RGB2BGR))
    # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    # # dbg_gray_path = str(frame_indices[0]) + '_gray.png'
    # # cv2.imwrite(dbg_gray_path, prev_frame)
    # for ii in range(1, num_frames):
    #     frame = _get_standard_frame(video, frame_indices[ii])
    #     # dbg_rgb_path = str(frame_indices[ii]) + '_rgb.png'
    #     # cv2.imwrite(dbg_rgb_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #     # dbg_gray_path = str(frame_indices[ii]) + '_gray.png'
    #     # cv2.imwrite(dbg_gray_path, frame)
    #     stack_of_diffs[:, :, ii - 1] = frame.astype(
    #         np.int32) - prev_frame.astype(np.int32)
    #     # dbg_diff_path = str(frame_indices[ii]) + '_diff.png'
    #     # cv2.imwrite(dbg_diff_path, np.squeeze(stack_of_diffs[:, :, ii - 1]))
    #     prev_frame = frame

    return stack_of_diffs


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
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')
    except:
        traceback.print_exc()
        sys.stderr.write('Error: failed to read video %s\n' % video_path)
        print('Error: failed to read video %s\n' % video_path)

    total_num_frames = video._meta['nframes']
    if total_num_frames <= 60:
        sys.sderr.write('Video %s is too small\n' % video_path)
        print('Video %s is too small\n' % video_path)
        return None, None
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

    # Generate center frames and stacks of differences
    center_frames_indices = map(lambda x: (x[-1] + x[0]) // 2, chunks)
    center_frames = list(map(lambda x: _get_standard_frame(video, x),
                             center_frames_indices))
    # stacks_of_diffs = list(map(lambda x: _create_stack_of_diffs(video, x),
    #                            chunks))
    stacks_of_diffs = [None] * len(chunks)
    # FIXME: we need to augment the input either here or by redesign!
    for ii in range(len(chunks)):
        stacks_of_diffs[ii] = _create_stack_of_diffs(video, chunks[ii])

    center_frames = np.float32(center_frames)
    stacks_of_diffs = np.float32(stacks_of_diffs)
    class_labels = np.int32(-1)
    path_no_ext, _ = osp.splitext(video_path)
    filenames = ['%s-frame_%d' % (path_no_ext, x) for x in
                 center_frames_indices]
# FIXME: return the correct class label!!
    return center_frames, stacks_of_diffs, [class_labels] * len(chunks), \
        filenames


def _shufle_motion():
    return None


def unit_test_supervised_input_for_train():
    # TODO: organize these constants and/or use variables!
    # TODO: organize unit tests, possibly in another file!
    data_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucf101'
    input_list_filepath = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/tmp_list.txt'
    batch_size = 4
    supervision_mode = 'supervised'
    run_mode = 'train'
    log_root = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/tf_logs'
    output_dump_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/outputs'

    center_frames_op, motion_encoding_op, class_label_op, filenames_op = \
        build_input(data_path, input_list_filepath, batch_size,
                    supervision_mode, run_mode, log_root, num_threads=16)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    num_batches = 4
    for batch_i in range(num_batches):
        (center_frames, motion_encodings, class_labels, filenames) = sess.run(
            [center_frames_op, motion_encoding_op, class_label_op,
             filenames_op])

        print('Storing sample output in ' + output_dump_path)
        for i in xrange(batch_size):
            print('Storing output for batch #%d, sample #%d (total: %d)' % (
                  batch_i, i, batch_i*batch_size + i + 1))
            if batch_size > 1:
                basename, ext = osp.splitext(osp.basename(filenames[i]))
            else:
                basename, ext = osp.splitext(osp.basename(filenames))

            with open(osp.join(output_dump_path, basename + '.pkl'), 'wb') as f:
                pickle.dump((center_frames, motion_encodings, class_labels,
                             filenames), f)


def unit_test_split_into_train_tuples():
    video_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucf101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
    output_dump_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/outputs'
    print('Splitting video %s' % video_path)
    centers, motion_encs, class_labels, paths_tiled = _split_into_train_tuples(video_path)

    print('class_labels = ', class_labels)
    print('paths_tiled = ', paths_tiled)
    for i in range(len(centers)):
        img = centers[i]
        center_filename = 'center_%d' % i
        center_filepath = osp.join(output_dump_path, center_filename)
        # b, g, r = cv2.split(img)
        # img = cv2.merge([r, g, b])
        print('saving center to %s' % center_filepath)
        fig = plt.figure()
        plt.imshow(img)
        fig.savefig(center_filepath, format='png')

        stack = motion_encs[i]
        stack = (stack + 255) / 2
        for j in range(stack.shape[-1]):
            motion_filename = 'center_%d_motion_%d' % (i, j)
            motion_filepath = osp.join(output_dump_path, motion_filename)
            print('saving motion to %s' % motion_filepath)
            fig = plt.figure()
            plt.imshow(stack[:, :, j], cmap='gray')
            fig.savefig(motion_filepath, format='png')


def _visualize_saved_pickle(pkl_file_path):
    """
    Visualize center frames and their stack of differences for some video.
    Args:
        pkl_file_path: path the generated pickle file for a video.
    """
    with open(pkl_file_path, 'rb') as f:
        centers, stacks_of_diffs, class_labels, filename = pickle.load(f)
    print('filename = %s -- class_label = %d' % (filename, class_labels))
    for i in range(len(centers)):
        img = centers[i]
        # b, g, r = cv2.split(img)
        # img = cv2.merge([r, g, b])
        plt.imshow(img)
        plt.show()

        stack = stacks_of_diffs[i]
        stack = (stack + 255) / 2
        for j in range(stack.shape[-1]):
            plt.imshow(stack[:, :, j], cmap='gray')
            plt.show()


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


def build_input(data_path, input_list_filepath, batch_size, supervision_mode,
                run_mode, log_root, num_threads=16):

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


if __name__ == '__main__':
    unit_test_supervised_input_for_train()
    # unit_test_split_into_train_tuples()
