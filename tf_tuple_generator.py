import argparse
import configuration as file_const
import constants as const
import cv2
import imageio
import math
import numpy as np
import os.path as osp
import pickle
import re
import time
import traceback
import tensorflow as tf
import tuple_generator_utils as gen_utils
import sys


global class_names_all


# deprecated: was used in the queue-based input pipeline
def _decode(filename_tf, cfg):
    input_video_path = tf.string_join([[cfg.data_path + '/'], filename_tf])
    input_video_path = tf.reshape(input_video_path, [-1])
    center_frames, motion_enc, class_labels, filenames_tiled = tf.py_func(
        _split_into_supervised_train_tuples, [input_video_path[0]], [
        tf.float32, tf.float32, tf.int32, tf.string])

    return center_frames, motion_enc, class_labels, filenames_tiled


# TODO: num_frames should be set/linked to const.context_frames + 1 to allow
# changing it from one place
def _split_into_supervised_train_tuples(video_path, num_frames=6):
    """
    Splits a video into consecutive but non-overlapping sequences.

    Args:
        video_path (str): path to source video (e.g. *.avi).
        num_frames (int): size of each sequence.

    Returns:
        center_frames: list of center frames for each sampled sequence. Each
            frame is a 3D array (height x widht x 3).
        stacks_of_diffs: list of stack_of_diffs for each sampled sequence. Each
            stack is a 3D array (height x width x num_frames-1).
    """
    global class_names_all

    video_path = video_path.decode("utf-8")
    video_pkl_path = video_path[:-4] + '.pkl'
    with open(video_pkl_path, 'rb') as f:
        frames = pickle.load(f)

    if len(frames) < 11:  # Too short to create even one tuple!
        return [], [], [], []

    # TODO: how many tuples to sample? currently every k_sampling_step
    k_sampling_step = 2  # this is to be consistent with 3 fps downsampling
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
        chunk_frames = list(map(lambda x: frames[x], chunks[tuple_i]))
        stacks_of_diffs[tuple_i], crop_coords = \
            gen_utils.create_stack_of_diffs_from_frames(
            chunk_frames, augment_flag=True)  # TODO: augment only for training
        center_frames[tuple_i] = gen_utils.standardize_frame(
            frames[center_frames_indices[tuple_i]], crop_coords)

    center_frames = np.float32(center_frames)
    stacks_of_diffs = np.float32(stacks_of_diffs)

    path_delimiter_regex = "[/\\\]"  # split on '/' or '\' for unix or windows
    class_name = re.compile(path_delimiter_regex).split(video_path)[-2]
    class_label = np.int32(class_names_all.index(class_name))

    path_no_ext, _ = osp.splitext(video_path)
    basename = osp.basename(path_no_ext)
    filenames = ['%s-frame_%d' % (basename, x) for x in
                 center_frames_indices]

    # class_labels = [[class_label]] * len(chunks)
    class_label_hot_vec = np.zeros(file_const.num_classes, dtype=np.int32)
    class_label_hot_vec[class_label] = 1
    class_labels_hot_vec = [class_label_hot_vec] * len(chunks)

    return center_frames, stacks_of_diffs, class_labels_hot_vec, filenames


# TODO: num_frames should be set/linked to const.context_frames + 1 to allow
# changing it from one place
def _split_into_unsupervised_train_tuples(vid_path1, vid_path2, num_frames=6):
    """
    Splits a video into consecutive but non-overlapping sequences.

    Args:
        video_path (str): path to source video (e.g. *.avi).
        num_frames (int): size of each sequence.

    Returns:
        center_frames: list of center frames for each sampled sequence. Each
            frame is a 3D array (height x widht x 3).
        stacks_of_diffs: list of stack_of_diffs for each sampled sequence. Each
            stack is a 3D array (height x width x num_frames-1).
    """

    path_no_ext1, _ = osp.splitext(vid_path1)
    path_no_ext2, _ = osp.splitext(vid_path2)
    basename1 = osp.basename(path_no_ext1)
    basename2 = osp.basename(path_no_ext2)
    vid_subject1 = basename1[:-3]
    vid_subject2 = basename2[:-3]
    if vid_subject1 == vid_subject2:  # should only operate on 2 different videos
        return [], [], [], []

    vid_path1 = vid_path1.decode("utf-8")
    vid_pkl_path1 = vid_path1[:-4] + '.pkl'
    with open(vid_pkl_path1, 'rb') as f:
        frames1 = pickle.load(f)

    vid_path2 = vid_path2.decode("utf-8")
    vid_pkl_path2 = vid_path2[:-4] + '.pkl'
    with open(vid_pkl_path2, 'rb') as f:
        frames2 = pickle.load(f)

    if len(frames1) < 11:  # Too short to create even one tuple!
        return [], [], [], []

    # TODO: how many tuples to sample? currently every k_sampling_step
    k_sampling_step = 2  # this is to be consistent with 3 fps downsampling
    chunks = []
    for idx in range(0, len(frames1), k_sampling_step):
        chunk = np.arange(
            idx, idx + (num_frames - 1) * k_sampling_step + 1, k_sampling_step)
        if chunk[-1] < len(frames1):
            chunks.append(chunk)
        else:
            break

    center_frames_indices = list(map(lambda x: (x[-1] + x[0]) // 2, chunks))
    # print('DBG1: ', center_frames_indices)
    # print('DBG2: ', chunks)
    num_tuples = len(center_frames_indices)
    center_frames = [None] * num_tuples
    stacks_of_diffs = [None] * num_tuples
    class_labels_hot_vec = list(np.zeros((num_tuples, 4)))
    for tuple_i in range(num_tuples):
        chunk_frames = list(map(lambda x: frames1[x], chunks[tuple_i]))
        stacks_of_diffs[tuple_i], crop_coords = \
            gen_utils.create_stack_of_diffs_from_frames(
            chunk_frames, augment_flag=True)  # TODO: augment only for training
        lbl = np.random.randint(4)
        class_labels_hot_vec[tuple_i][lbl] = 1
        if lbl == 0:  # correct image and correct motion
            center_frames[tuple_i] = gen_utils.standardize_frame(
                frames1[center_frames_indices[tuple_i]], crop_coords)
        elif lbl == 1:  # correct image, but shuffled motion
            center_frames[tuple_i] = gen_utils.standardize_frame(
                frames1[center_frames_indices[tuple_i]], crop_coords)
            stacks_of_diffs[tuple_i] = _shufle_motion(stacks_of_diffs[tuple_i])
        elif lbl == 2:  # incorrect image, but correct motion
            center_frames[tuple_i] = gen_utils.standardize_frame(
                frames2[center_frames_indices[tuple_i] % len(frames2)], None)
        else:  # incorrect image and shuffled motion
            center_frames[tuple_i] = gen_utils.standardize_frame(
                frames2[center_frames_indices[tuple_i] % len(frames2)], None)
            stacks_of_diffs[tuple_i] = _shufle_motion(stacks_of_diffs[tuple_i])

    center_frames = np.float32(center_frames)
    stacks_of_diffs = np.float32(stacks_of_diffs)
    class_labels_hot_vec = np.int32(class_labels_hot_vec)

    path_no_ext, _ = osp.splitext(vid_path1)
    basename = osp.basename(path_no_ext)
    filenames = ['%s-frame_%d' % (basename, x) for x in
                 center_frames_indices]

    return center_frames, stacks_of_diffs, class_labels_hot_vec, filenames


# # This is the old split code for the queue-base input pipeline
# def _split_into_supervised_train_tuples(video_path, num_frames=6, step=15,
#                                         sampling_step=2):
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


def _is_valid_perm(arr):
    """
    Checks if arr represents a valid permutation to produce a negative motion
    typle.
    A valid permutation is one that is not in ascending or descending order.

    Args:
        arr: 1D array of a permuted sequence of numbers.

    Returns:
        True or False indicating whether the input is a valid negative tuple
        permutation.
    """

    diff = arr[1] - arr[0]
    for i in range(2, len(arr)):
        if arr[i] - arr[i-1] != diff:
            return True  # correctly shuffled to be in a wrong order

    return False


def _shufle_motion(motion_enc):
    """
    shuffles encoded frames (e.g. stack_of_diffs) to produce a negative tuple.
    It ensures that the suffled motion is not in ascending nor descending order.

    Args:
        motion_enc (3D array): W x H x num_channles array representing the
            encoded motion (e.g. stack of difference representation)

    Returns:
        shuffled_motion_enc: 3D array of the same dimensions but after shuffling
            the last dimension.
            frame is a 3D array (height x widht x 3).
        stacks_of_diffs: list of stack_of_diffs for each sampled sequence. Each
            stack is a 3D array (height x width x num_frames-1).
    """

    num_channels = motion_enc.shape[2]
    perm = np.random.permutation(num_channels)
    while not _is_valid_perm(perm):
        perm = np.random.permutation(num_channels)

    return motion_enc[:, :, perm]


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
    # inp_imgs_tf = tf.reshape(inp_imgs_tf, [-1, img_height, img_width, 3])
    # motion_encs_tf = tf.reshape(motion_encs_tf, [-1, img_height, img_width, 
    #                                              num_motion_channels])
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

    # Randomly choose 1 of 4 random ordering for chromatic changes OR none
    order_rand = tf.random_uniform(shape=[], minval=0, maxval=8)

    inp_imgs_tf = tf.case(
        {tf.less(order_rand, 1):
            # lambda: tf.map_fn(lambda image: order_1(image), inp_imgs_tf),
            lambda: order_1(inp_imgs_tf),
         tf.less(order_rand, 2):
            # lambda: tf.map_fn(lambda image: order_2(image), inp_imgs_tf),
            lambda: order_2(inp_imgs_tf),
         tf.less(order_rand, 3):
            # lambda: tf.map_fn(lambda image: order_3(image), inp_imgs_tf),
            lambda: order_3(inp_imgs_tf),
         tf.less(order_rand, 4):
            # lambda: tf.map_fn(lambda image: order_4(image), inp_imgs_tf)
            lambda: order_3(inp_imgs_tf)
        },
        # default=lambda: tf.map_fn(lambda image: order_4(image), inp_imgs_tf))
        # default=lambda: order_4(inp_imgs_tf))
        default=lambda: inp_imgs_tf)

    return inp_imgs_tf, motion_encs_tf


def _build_supervised_input_for_train(filenames_dataset, augmentation_flag,
                                      buffer_size, batch_size, cfg):
    frame_height = cfg['frame_height']
    frame_width = cfg['frame_width']
    context_channels = cfg['context_channels']

    # generates multiple tuples from each input video file
    dataset = filenames_dataset.map(
        lambda filename: tuple(tf.py_func(
            _split_into_supervised_train_tuples, [filename], [
            tf.float32, tf.float32, tf.int32, tf.string])))

    # flattens the multiple tuples from all videos into a single list of tuples.
    dataset = dataset.flat_map(
        lambda center_frames, stacks_of_diffs, class_labels, filenames:
            tf.data.Dataset.from_tensor_slices((
                center_frames, stacks_of_diffs, class_labels, filenames)))

    # TODO: I'm pretty sure you can define the shapes when creating the tensors
    # instead of this dummy reshape! So, implement this change, please!
    def _dummy_reshape_map(center_frame, stack_of_diffs, class_label, filename):
        # center_frame = tf.reshape(center_frame, [frame_height, frame_width, 3])
        # stack_of_diffs = tf.reshape(stack_of_diffs, [frame_height, frame_width,
        #                                              context_channels])
        center_frame.set_shape([frame_height, frame_width, 3])
        stack_of_diffs.set_shape([frame_height, frame_width, context_channels])
        return center_frame, stack_of_diffs, class_label, filename

    # dummy reshape to make tensor shapes known for two_stream.py assertions
    dataset = dataset.map(_dummy_reshape_map)

    # Shuffle training examples and create mini-batch
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    # Data augmentation
    def augment_wrapper(center_frames, stacks_of_diffs, class_labels,
                        filenames):
        center_frames, stacks_of_diffs = augment_data(
            center_frames, stacks_of_diffs, frame_height,
            frame_width, context_channels)
        return center_frames, stacks_of_diffs, class_labels, filenames

    # FIXME: apply augmentation per image, not per mini-batch, unit-test that.
    if augmentation_flag:
        dataset = dataset.map(augment_wrapper)

    # FIXME: the problem with image-based augmentation is a lot slower
    # # Shuffle training examples and create mini-batch
    # dataset = dataset.shuffle(buffer_size=buffer_size)
    # dataset = dataset.batch(batch_size)

    return dataset


# Deprecated
# still "sometimes" crashes. seems like still a problem with reading
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


def _build_unsupervised_input_for_train(filenames_dataset, augmentation_flag,
                                        buffer_size, batch_size, cfg):
    frame_height = cfg['frame_height']
    frame_width = cfg['frame_width']
    context_channels = cfg['context_channels']

    # generates multiple tuples from each input video file
    filenames_dataset2 = filenames_dataset.shuffle(buffer_size=1000)
    paired_dataset = tf.data.Dataset.zip((filenames_dataset,
                                          filenames_dataset2))
    dataset = paired_dataset.map(
        lambda filename1, filename2: tuple(tf.py_func(
            _split_into_unsupervised_train_tuples, [filename1, filename2], [
            tf.float32, tf.float32, tf.int32, tf.string])))

    # flattens the multiple tuples from all videos into a single list of tuples.
    dataset = dataset.flat_map(
        lambda center_frames, stacks_of_diffs, class_labels, filenames:
            tf.data.Dataset.from_tensor_slices((
                center_frames, stacks_of_diffs, class_labels, filenames)))

    # TODO: I'm pretty sure you can define the shapes when creating the tensors
    # instead of this dummy reshape! So, implement this change, please!
    def _dummy_reshape_map(center_frame, stack_of_diffs, class_label, filename):
        # center_frame = tf.reshape(center_frame, [frame_height, frame_width, 3])
        # stack_of_diffs = tf.reshape(stack_of_diffs, [frame_height, frame_width,
        #                                              context_channels])
        center_frame.set_shape([frame_height, frame_width, 3])
        stack_of_diffs.set_shape([frame_height, frame_width, context_channels])
        return center_frame, stack_of_diffs, class_label, filename

    # dummy reshape to make tensor shapes known for two_stream.py assertions
    dataset = dataset.map(_dummy_reshape_map)

    # Shuffle training examples and create mini-batch
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    # Data augmentation
    def augment_wrapper(center_frames, stacks_of_diffs, class_labels,
                        filenames):
        center_frames, stacks_of_diffs = augment_data(
            center_frames, stacks_of_diffs, frame_height,
            frame_width, context_channels)
        return center_frames, stacks_of_diffs, class_labels, filenames

    # FIXME: apply augmentation per image, not per mini-batch, unit-test that.
    if augmentation_flag:
        dataset = dataset.map(augment_wrapper)

    # FIXME: the problem with image-based augmentation is a lot slower
    # # Shuffle training examples and create mini-batch
    # dataset = dataset.shuffle(buffer_size=buffer_size)
    # dataset = dataset.batch(batch_size)

    return dataset


def _build_unsupervised_input_for_test():
    return None, None, None, None


def build_input(data_path, input_list_filepaths, activities_list_path,
                batch_size, supervision_mode, run_mode):

    global class_names_all

    cfg = dict()
    cfg['frame_height'] = const.frame_height
    cfg['frame_width'] = const.frame_width
    cfg['context_channels'] = const.context_channels
    buffer_size = const.buffer_size  # TODO: rethink size (currently 5000)

    with open(activities_list_path, 'r') as f:
        class_names_all = [x.strip() for x in f.readlines()]

    # dataset of file names
    dataset = tf.data.TextLineDataset(input_list_filepaths)
    dataset = dataset.shuffle(buffer_size=1000)

    # dataset of file paths
    dataset = dataset.map(
        lambda filename: tf.string_join([data_path + '/', filename]))

    if supervision_mode == 'supervised':
        if run_mode == 'train':
            dataset = _build_supervised_input_for_train(
                dataset, True, buffer_size, batch_size, cfg)
        elif run_mode == 'test':
            # dataset = _build_supervised_input_for_test()
            raise NotImplementedError  # TODO
        else:
            raise ValueError('Unknow run_mode: %s' % run_mode)
    elif supervision_mode == 'unsupervised':
        if run_mode == 'train':
            dataset = _build_unsupervised_input_for_train(
                dataset, True, buffer_size, batch_size, cfg)
        elif run_mode == 'test':
            # dataset = _build_unsupervised_input_for_test()
            raise NotImplementedError  # TODO
        else:
            raise ValueError('Unknow run_mode: %s' % run_mode)
    else:
        raise ValueError('Unknow supervision_mode: %s' % supervision_mode)

    # dataset = dataset.prefetch(const.buffer_size)

    return dataset
