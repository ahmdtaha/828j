# from easydict import EasyDict as edict
import argparse
import constants as const
import cv2
import imageio
import math
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

    video_path = video_path.decode("utf-8")
    video_pkl_path = video_path[:-4] + '.pkl'
    k_sampling_step = 2
    with open(video_pkl_path, 'rb') as f:
        frames = pickle.load(f)

    if len(frames) < 11:  # Too short to create even one tuple!
        return [], [], [], []

    # TODO: how many tuples to sample? currently  every k_sampling_step
    chunks = []
    for idx in range(0, len(frames), k_sampling_step):
        chunk = np.arange(
            idx, idx + (num_frames - 1) * k_sampling_step + 1, k_sampling_step)
        if chunk[-1] < len(frames):
            chunks.append(chunk)
        else:
            break

    center_frames_indices = list(map(lambda x: (x[-1] + x[0]) // 2, chunks))
    # print('DBG1: ', center_frames_indices)
    # print('DBG2: ', chunks)
    num_tuples = len(center_frames_indices)
    center_frames = [None] * num_tuples
    stacks_of_diffs = [None] * num_tuples
    for tuple_i in range(num_tuples):
        center_frames[tuple_i] = gen_utils.standardize_frame(
            frames[center_frames_indices[tuple_i]])
        chunk_frames = list(map(lambda x: frames[x], chunks[tuple_i]))
        stacks_of_diffs[tuple_i] = gen_utils.create_stack_of_diffs_from_frames(
            chunk_frames, augment_flag=True)  # TODO: augment only for training

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


# def _split_into_train_tuples(video_path, num_frames=6, step=15,
#                              sampling_step=2):
#     """
#     Splits a video into consecutive but non-overlapping sequences.
# 
#     Args:
#         video_path (str): path to source video (e.g. *.avi).
#         num_frames (int): size of each sequence.
#         step (int): "Default" step between frames within a sampled sequence. It
#             can be readjusted for short videos.
# 
#     Returns:
#         center_framess: list of center frames for each sampled sequence. Each
#             frame is a 3D array (height x widht x 3).
#         stacks_of_diffs: list of stack_of_diffs for each sampled sequence. Each
#             stack is a 3D array (height x width x num_frames-1).
#     """
#     global class_names_all
# 
#     try:
#         video = imageio.get_reader(video_path, 'ffmpeg')
#     except:
#         traceback.print_exc()
#         sys.stderr.write('Error: failed to read video %s\n' % video_path)
#         print('Error: failed to read video %s\n' % video_path)
#         return
# 
#     total_num_frames = video._meta['nframes']
#     # FIXME: if you want to exclude short videos, remove them as preprocessing!
#     # if total_num_frames <= 60:
#     #     sys.stderr.write('Video %s is too small\n' % video_path)
#     #     print('Video %s is too small\n' % video_path)
#     #     return [], [], [], []
#     if total_num_frames <= 60:
#         step = total_num_frames // (num_frames - 1)
#     elif total_num_frames <= 75:  # then re-adjust step
#         if total_num_frames >= 71:
#             step = 14
#         elif total_num_frames >= 66:
#             step = 13
#         else:
#             step = 12
# 
#     chunks = []
#     for start_frame in range(0, total_num_frames, sampling_step):
#         chunk = np.arange(
#             start_frame, start_frame + (num_frames - 1) * step + 1, step)
#         if chunk[-1] < total_num_frames:
#             chunks.append(chunk)
#             start_frame = chunk[-1] + 1
#         else:
#             # # if remaining frames cannot complete a chunk, then center
#             # # previous chunks so that leftovers are equally two-sided
#             # shift = (total_num_frames - start_frame + 1) // 2
#             # for i in range(len(chunks)):
#             #     chunks[i] = chunks[i] + shift
#             break
# 
#     center_frames_indices = map(lambda x: (x[-1] + x[0]) // 2, chunks)
#     center_frames, stacks_of_diffs = gen_utils.split_into_tuples(
#         video_path, chunks, augment_tuples=True)
# 
#     center_frames = np.float32(center_frames)
#     stacks_of_diffs = np.float32(stacks_of_diffs)
#     # FIXME: next line assumes unix paths (won't work for windows). Use regex
#     # to split on both / and \
#     class_name = video_path.split('/')[-2]
#     class_label = np.int32(class_names_all.index(class_name))
#     path_no_ext, _ = osp.splitext(video_path)
#     filenames = ['%s-frame_%d' % (path_no_ext, x) for x in
#                  center_frames_indices]
# 
#     return center_frames, stacks_of_diffs, [class_label] * len(chunks), \
#         filenames
# 
# 
def _shufle_motion():
    return None


# Data augmentation
def augment_data(inp_imgs_tf, motion_encs_tf, img_height, img_width,
                 num_motion_channels):

    # Flip horizontally
    hor_flip_bool = tf.less(tf.random_uniform(shape=[], minval=0, maxval=1.),
                            0.5)
    inp_imgs_tf, motion_encs_tf = tf.cond(
        hor_flip_bool, lambda: (tf.image.flip_left_right(inp_imgs_tf),
                                tf.image.flip_left_right(motion_encs_tf)),
                       lambda: (inp_imgs_tf, motion_encs_tf))

    # TODO: should we do vertical flipping?
    # Flip vertically
    ver_flip_bool = tf.less(tf.random_uniform(shape=[], minval=0, maxval=1.),
                            0.5)
    inp_imgs_tf, motion_encs_tf = tf.cond(
        ver_flip_bool, lambda: (tf.image.flip_up_down(inp_imgs_tf),
                                tf.image.flip_up_down(motion_encs_tf)),
                       lambda: (inp_imgs_tf, motion_encs_tf))

    # dummy reshape operation because tf.contrib.image.transform operation
    # needs to know the tensorshape beforehand!
    # FIXME: I suspect this dummy reshape causes vertical flipping, possibly
    # because height = width = 227? That or either some input videos are
    # vertically flipped
    inp_imgs_tf = tf.reshape(inp_imgs_tf, [-1, img_height, img_width, 3])
    motion_encs_tf = tf.reshape(motion_encs_tf, [-1, img_height, img_width, 
                                                 num_motion_channels])
    # inp_imgs_tf = tf.reshape(inp_imgs_tf, [img_height, img_width, 3])
    # motion_encs_tf = tf.reshape(motion_encs_tf, [img_height, img_width, 
    #                                              num_motion_channels])

    # Random rotation
    rotate_flag = tf.less(tf.random_uniform(shape=[], minval=0, maxval=1.), 0.7)

    # angle = tf.random_uniform(shape=[], maxval=2 * math.pi)
    max_angle = 30.0 * math.pi / 180
    angle = tf.random_uniform(shape=[], maxval=2 * max_angle) - max_angle

    inp_imgs_tf, motion_encs_tf = tf.cond(
        rotate_flag, lambda: (tf.contrib.image.rotate(inp_imgs_tf, angle),
                              tf.contrib.image.rotate(motion_encs_tf, angle)),
                     lambda: (inp_imgs_tf, motion_encs_tf))

    # angle = 45 * math.pi / 180
    # inp_imgs_tf = tf.contrib.image.rotate(inp_imgs_tf, angle)
    # motion_encs_tf = tf.contrib.image.rotate(motion_encs_tf, angle)

    # Chromatic chages

    # Color distortions are non-commutative, and so the order matters
    def order_1(img):
        img = tf.image.random_brightness(img, max_delta=32./255.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        return img

    def order_2(img):
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_brightness(img, max_delta=32./255.)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        return img

    def order_3(img):
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_brightness(img, max_delta=32./255.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        return img

    def order_4(img):
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.image.random_brightness(img, max_delta=32./255.)
        return img

    # Randomly choose 1 of 4 random ordering for chromatic changes
    order_rand = tf.random_uniform(shape=[], minval=0, maxval=4)

    inp_imgs_tf = tf.case(
        {tf.less(order_rand, 1):
            lambda: tf.map_fn(lambda image: order_1(image), inp_imgs_tf),
            # lambda: order_1(inp_imgs_tf),
         tf.less(order_rand, 2):
            lambda: tf.map_fn(lambda image: order_2(image), inp_imgs_tf),
            # lambda: order_2(inp_imgs_tf),
         tf.less(order_rand, 3):
            lambda: tf.map_fn(lambda image: order_3(image), inp_imgs_tf)
            # lambda: order_3(inp_imgs_tf)
        },
        default=lambda: tf.map_fn(lambda image: order_4(image), inp_imgs_tf))
        # default=lambda: order_4(inp_imgs_tf))

    return inp_imgs_tf, motion_encs_tf


def _build_supervised_input_for_train(filenames_tf, cfg, num_threads):
    center_frames, motion_encodings, class_labels, filenames_tf_tiled = _decode(
        filenames_tf, cfg)
    # TODO: make sure to cast read data to the correct dtype

    # Data augmentation
    if cfg.augmentation_flag:
        center_frames, motion_encodings = augment_data(
            center_frames, motion_encodings, cfg.img_height, cfg.img_width,
            cfg.context_channels)
        # None

    # TODO: check if values for the capacity and min_after_dequeue are not bad
    example_queue = tf.RandomShuffleQueue(
        capacity=4*cfg.batch_size, min_after_dequeue=2*cfg.batch_size,
        dtypes=[tf.float32, tf.float32, tf.int32, tf.string],
        shapes=[[cfg.img_height, cfg.img_width, cfg.img_channels],
                [cfg.img_height, cfg.img_width, cfg.context_channels], [], []])

    # print('DBG1: ', center_frames.get_shape())
    # print('DBG2: ', motion_encodings.get_shape())
    example_enqueue_op = example_queue.enqueue_many(
        [center_frames, motion_encodings, class_labels, filenames_tf_tiled])

    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_enqueue_op] * num_threads))

    # Read batch
    center_frame, motion_encoding, class_label, tuple_name = \
        example_queue.dequeue_many(cfg.batch_size)

    return center_frame, motion_encoding, class_label, tuple_name


# FIXME: still "sometimes" crashes. seems like still a problem with reading
# videos, but i'm not 100% sure yet. Maybe also it is because of the extra
# augmentations, as it "seems" to work fine if augmentation is disabled.
def _build_supervised_input_for_train_queues(filenames_tf, cfg, num_threads):
    center_frames, motion_encodings, class_labels, filenames_tf_tiled = _decode(
        filenames_tf, cfg)
    # TODO: make sure to cast read data to the correct dtype

    # Data augmentation
    if cfg.augmentation_flag:
        center_frames, motion_encodings = augment_data(
            center_frames, motion_encodings, cfg.img_height, cfg.img_width,
            cfg.context_channels)
        # None

    # TODO: check if values for the capacity and min_after_dequeue are not bad
    example_queue = tf.RandomShuffleQueue(
        capacity=4*cfg.batch_size, min_after_dequeue=2*cfg.batch_size,
        dtypes=[tf.float32, tf.float32, tf.int32, tf.string],
        shapes=[[cfg.img_height, cfg.img_width, cfg.img_channels],
                [cfg.img_height, cfg.img_width, cfg.context_channels], [], []])

    # print('DBG1: ', center_frames.get_shape())
    # print('DBG2: ', motion_encodings.get_shape())
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

    # cfg = edict()
    # cfg.data_path = data_path
    # cfg.batch_size = batch_size
    # cfg.supervision_mode = supervision_mode
    # cfg.run_mode = run_mode
    # cfg.log_root = log_root
    # cfg.augmentation_flag = True if run_mode == 'train' else False
    # cfg.img_height = const.frame_height
    # cfg.img_width = const.frame_width
    # cfg.img_channels = const.frame_channels
    # cfg.context_channels = const.context_channels
    augmentation_flag = True if run_mode == 'train' else False

    with open(activities_list_path, 'r') as f:
        class_names_all = [x.strip() for x in f.readlines()]

    # dataset of file names
    dataset = tf.data.TextLineDataset(input_list_filepath)
    print('DBG: ', dataset)

    # dataset of file paths
    dataset = dataset.map(
        lambda filename: tf.string_join([data_path + '/', filename]))
    print('DBG: ', dataset)

    # FIXME: you cannot make a python tuple out of multiple training tuples per video
    # decode each video into tuples
    dataset = dataset.map(
        lambda filename: tuple(tf.py_func(
            _split_into_train_tuples, [filename], [
            tf.float32, tf.float32, tf.int32, tf.string])))
    print('DBG: ', dataset)

    # FIXME: rename variables if this checked out
    dataset = dataset.flat_map(
        lambda center_frames, stacks_of_diffs, class_labels, filenames:
            tf.data.Dataset.from_tensor_slices((
                center_frames, stacks_of_diffs, class_labels, filenames)))
    print('DBG: ', dataset)

    # Shuffle training examples and create mini-batch
    dataset = dataset.shuffle(buffer_size=5000)  # FIXME TODO: rethink buffer_size
    dataset = dataset.batch(batch_size)

    # Data augmentation
    def augment_aux(center_frames, stacks_of_diffs, class_labels, filenames):
        center_frames, stacks_of_diffs = augment_data(
            center_frames, stacks_of_diffs, const.frame_height,
            const.frame_width, const.context_channels)
        return center_frames, stacks_of_diffs, class_labels, filenames

    if augmentation_flag:
        dataset = dataset.map(augment_aux)

    return dataset

# ==============================================================================

    # filename_queue = tf.train.string_input_producer([input_list_filepath])
    # reader = tf.TextLineReader()
    # _, filenames_tf = reader.read(filename_queue)

    # if supervision_mode == 'supervised':
    #     if run_mode == 'train' or run_mode == 'eval':
    #         center_frame, motion_encoding, class_label, tuple_name = \
    #            _build_supervised_input_for_train(filenames_tf, cfg, num_threads)
    #     elif run_mode == 'test':
    #         # center_frame, motion_encoding, tuple_name, class_label = \
    #         #     _build_supervised_input_for_test()
    #         raise NotImplementedError  # TODO
    #     else:
    #         raise ValueError('Unknow run_mode: %s' % run_mode)
    # elif supervision_mode == 'unsupervised':
    #     if run_mode == 'train' or run_mode == 'eval':
    #         # center_frame, motion_encoding, tuple_name, class_label = \
    #         #     _build_unsupervised_input_for_train()
    #         raise NotImplementedError  # TODO
    #     elif run_mode == 'test':
    #         # center_frame, motion_encoding, tuple_name, class_label = \
    #         #     _build_unsupervised_input_for_test()
    #         raise NotImplementedError  # TODO
    #     else:
    #         raise ValueError('Unknow run_mode: %s' % run_mode)
    # else:
    #     raise ValueError('Unknow supervision_mode: %s' % supervision_mode)

    # return center_frame, motion_encoding, class_label, tuple_name
