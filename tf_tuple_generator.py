import configuration as file_const
import constants as const
import math
import numpy as np
import os.path as osp
import pickle
import re
import tensorflow as tf
import tuple_generator_utils as gen_utils


global class_names_all


def _split_into_supervised_tuples(video_path, num_frames=6,
                                  augmentation_flag=False):
    """
    Splits a video into consecutive but non-overlapping tuples / sequences.

    Args:
        video_path (str): path to source video (e.g. *.avi).
        num_frames (int): size of each tuple / sequence.
        augmentation_flag (bool): whether to augment tuples or not (e.g.
            cropping, channel-splitting, ... etc)

    Returns:
        center_frames: list of center frames for each sampled sequence. Each
            frame is a 3D array (height x widht x 3).
        stacks_of_diffs: list of stack_of_diffs for each sampled sequence. Each
            stack is a 3D array (height x width x num_frames-1).
    """

    global class_names_all


    ## dataset_path
    video_path = video_path.decode("utf-8")
    video_pkl_path = video_path[:-4] + '.pkl'
    with open(video_pkl_path, 'rb') as f:
        frames = pickle.load(f)

    if len(frames) < 11:  # Too short to create even one tuple!
        pass

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
            chunk_frames, augment_flag=augmentation_flag)
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


def _split_into_unsupervised_tuples(vid_path1, vid_path2, num_frames=6,
                                    augmentation_flag=False):
    """
    Splits a video into tuples for unsupervised training (4 classes spanning
        correct/incorrect spatial frame X correct/shuffled motion encoding).

    Args:
        vid_path1 (str): path to source video (e.g. *.avi).
        vid_path2 (str): path to a 2nd video from which to use incorrect
            spatial frames.
        num_frames (int): size of each tuple / sequence.
        augmentation_flag (bool): whether to augment tuples or not (e.g.
            cropping, channel-splitting, ... etc)

    Returns:
        center_frames: list of center frames for each sampled sequence. Each
            frame is a 3D array (height x widht x 3). Each center frame can
            be either the correct frame (from video1) corresponing to the
            motion, or an incorrect frame (from video2).
        stacks_of_diffs: list of stack_of_diffs for each sampled sequence. Each
            stack is a 3D array (height x width x num_frames-1).
    """

    path_no_ext1, _ = osp.splitext(vid_path1)
    path_no_ext2, _ = osp.splitext(vid_path2)
    basename1 = osp.basename(path_no_ext1)
    basename2 = osp.basename(path_no_ext2)
    vid_subject1 = basename1[:-3]
    vid_subject2 = basename2[:-3]
    # Should skip pair of videos that belong to the same human subject & action
    if vid_subject1 == vid_subject2:
        pass

    vid_path1 = vid_path1.decode("utf-8")
    vid_pkl_path1 = vid_path1[:-4] + '.pkl'
    with open(vid_pkl_path1, 'rb') as f:
        frames1 = pickle.load(f)

    vid_path2 = vid_path2.decode("utf-8")
    vid_pkl_path2 = vid_path2[:-4] + '.pkl'
    with open(vid_pkl_path2, 'rb') as f:
        frames2 = pickle.load(f)

    if len(frames1) < 11:  # Too short to create even one tuple!
        pass

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
    num_tuples = len(center_frames_indices)
    center_frames = [None] * num_tuples
    stacks_of_diffs = [None] * num_tuples
    class_labels_hot_vec = list(np.zeros((num_tuples, 4)))
    for tuple_i in range(num_tuples):
        chunk_frames = list(map(lambda x: frames1[x], chunks[tuple_i]))
        stacks_of_diffs[tuple_i], crop_coords = \
            gen_utils.create_stack_of_diffs_from_frames(
            chunk_frames, augment_flag=augmentation_flag)
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


def _is_valid_perm(arr):
    """
    Checks if arr represents a valid permutation to produce a negative motion
    tuple. A valid permutation is one that is NOT in ascending or descending
    order.

    Args:
        arr: 1D array of a permuted sequence of integers.

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
    Shuffles encoded frames (e.g. stack_of_diffs) to produce a negative tuple.
    It ensures that the suffled motion is not in ascending nor descending order

    Args:
        motion_enc (3D array): W x H x num_channles array representing the
            encoded motion (e.g. stack of difference representation)

    Returns:
        shuffled_motion_enc: 3D array of the same dimensions but after
            shuffling the last dimension.
            frame is a 3D array (height x widht x 3).
    """

    num_channels = motion_enc.shape[2]
    perm = np.random.permutation(num_channels)
    while not _is_valid_perm(perm):
        perm = np.random.permutation(num_channels)

    return motion_enc[:, :, perm]


# Data augmentation
def augment_data(inp_imgs_tf, motion_encs_tf):
    """
    Performs more data augmentation like horizonatal and vertical flipping,
    rotation, chromatic distrotions.

    Args:
        inp_imgs_tf: either a single image tensor or a mini-batch of images.
        motion_encs_tf: either a single motion encoding (e.g. stack-of-diff) or
            a mini-batch of motion encodings.
    """

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

    # Random rotation
    rotate_flag = tf.less(tf.random_uniform(shape=[], minval=0, maxval=1.),
                          0.7)

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
    def _order_1(img):
        img = tf.image.random_brightness(img, max_delta=32./255.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        return img

    def _order_2(img):
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_brightness(img, max_delta=32./255.)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        return img

    def _order_3(img):
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_brightness(img, max_delta=32./255.)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        return img

    def _order_4(img):
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
        img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
        img = tf.image.random_brightness(img, max_delta=32./255.)
        return img

    # Randomly choose 1 of 4 random ordering for chromatic changes OR none
    order_rand = tf.random_uniform(shape=[], minval=0, maxval=8)

    inp_imgs_tf = tf.case(
        {tf.less(order_rand, 1):
            lambda: _order_1(inp_imgs_tf),
         tf.less(order_rand, 2):
            lambda: _order_2(inp_imgs_tf),
         tf.less(order_rand, 3):
            lambda: _order_3(inp_imgs_tf),
         tf.less(order_rand, 4):
            lambda: _order_4(inp_imgs_tf)
         },
        # default=lambda: order_4(inp_imgs_tf))
        default=lambda: inp_imgs_tf)

    return inp_imgs_tf, motion_encs_tf


def _build_input_aux(ragged_tuples_dataset, augmentation_flag, buffer_size,
                     batch_size, cfg):
    """
    Continues building supervised/unsupervised training tuples by taking in
    a dataset where each entry represents generated tuples from a single video,
    and then it flattens the dataset so that each entry is just one tuple,
    shuffles the tuples, sets the shapes of the tensors of each tuple,
    applies data-augmentation, and creates mini-batches.

    Args:
        ragged_tuples_dataset (tf.data.Dataset): a dataset where each entry
            represents generated tuples from a single video (videos of
            different lengths have different number of generated tuples)
        augmentation_flag (bool): whether to augment tuples or not (e.g.
            cropping, channel-splitting, ... etc)
        buffer_size: number of input tuples to buffer / pre-fetch for
            tf.data.Dataset shuffling for creating mini-batches.
        batch_size: size for mini-batch.
        cfg: dictionary representating input configurations such as frames'
            height and width, number of context channels, ... etc.

    Returns:
        dataset (tf.data.Dataset): tf dataset of tuples, each tuple consists of
            a center frame, motion-encoding (such as stack-of-difference),
            class-label, and file-name from which this tuple was generated.
    """

    frame_height = cfg['frame_height']
    frame_width = cfg['frame_width']
    context_channels = cfg['context_channels']

    # flattens the multiple tuples from all videos into a single list of tuples
    dataset = ragged_tuples_dataset.flat_map(
        lambda center_frames, stacks_of_diffs, class_labels, filenames:
            tf.data.Dataset.from_tensor_slices((
                center_frames, stacks_of_diffs, class_labels, filenames)))

    # Set tensor shapes to make thmem known for two_stream.py assertions
    def _set_shape_mapper(center_frame, stack_of_diffs, class_label, filename):
        center_frame.set_shape([frame_height, frame_width, 3])
        stack_of_diffs.set_shape([frame_height, frame_width, context_channels])
        return center_frame, stack_of_diffs, class_label, filename

    dataset = dataset.map(_set_shape_mapper)

    # Shuffle training examples and create mini-batch
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)

    # FIXME: apply augmentation per image, not per mini-batch, unit-test that.
    # Data augmentation
    def augment_wrapper(center_frames, stacks_of_diffs, class_labels,
                        filenames):
        center_frames, stacks_of_diffs = augment_data(
            center_frames, stacks_of_diffs)
        return center_frames, stacks_of_diffs, class_labels, filenames

    # FIXME: re-enable the extra augmentation!!
    if augmentation_flag and const.enable_extra_augmentation:
        dataset = dataset.map(augment_wrapper)

    # # Shuffle training examples and create mini-batch
    # dataset = dataset.shuffle(buffer_size=buffer_size)
    # dataset = dataset.batch(batch_size)

    return dataset


def _build_supervised_input(filenames_dataset, augmentation_flag, buffer_size,
                            batch_size, cfg):
    """
    Builds supervised input tuples for the TwoStream network.

    Args:
        filenames_dataset (tf.data.Dataset): dataset of input file paths, each
            entry represents a path to an input video (or its pickled
            representation).
        augmentation_flag (bool): whether to augment tuples or not (e.g.
            cropping, channel-splitting, ... etc)
        buffer_size: number of input tuples to buffer / pre-fetch for
            tf.data.Dataset shuffling for creating mini-batches.
        batch_size: size for mini-batch.
        cfg: dictionary representating input configurations such as frames'
            height and width, number of context channels, ... etc.

    Returns:
        dataset (tf.data.Dataset): tf dataset of tuples, each tuple consists of
            a center frame, motion-encoding (such as stack-of-difference),
            class-label, and file-name from which this tuple was generated.
    """

    context_channels = cfg['context_channels']

    # generates multiple tuples from each input video file
    dataset = filenames_dataset.map(
        lambda filename: tuple(tf.py_func(
            _split_into_supervised_tuples, [
                filename, context_channels+1, augmentation_flag], [
                tf.float32, tf.float32, tf.int32, tf.string])))

    return _build_input_aux(dataset, augmentation_flag, buffer_size,
                            batch_size, cfg)


def _build_unsupervised_input(filenames_dataset, augmentation_flag,
                              buffer_size, batch_size, cfg):
    """
    Builds unsupervised input tuples for the TwoStream network.

    Args:
        filenames_dataset (tf.data.Dataset): dataset of input file paths, each
            entry represents a path to an input video (or its pickled
            representation).
        augmentation_flag (bool): whether to augment tuples or not (e.g.
            cropping, channel-splitting, ... etc)
        buffer_size: number of input tuples to buffer / pre-fetch for
            tf.data.Dataset shuffling for creating mini-batches.
        batch_size: size for mini-batch.
        cfg: dictionary representating input configurations such as frames'
            height and width, number of context channels, ... etc.

    Returns:
        dataset (tf.data.Dataset): tf dataset of tuples, each tuple consists of
            a center frame, motion-encoding (such as stack-of-difference),
            class-label, and file-name from which this tuple was generated.
    """

    context_channels = cfg['context_channels']

    filenames_dataset2 = filenames_dataset.shuffle(buffer_size=1000)
    paired_dataset = tf.data.Dataset.zip((filenames_dataset,
                                          filenames_dataset2))
    # paired_dataset = paired_dataset.filter(lambda a, b: tf.not_equal(a, b))

    # generates multiple tuples from each input video file
    dataset = paired_dataset.map(
        lambda filename1, filename2: tuple(tf.py_func(
            _split_into_unsupervised_tuples, [
                filename1, filename2, context_channels+1, augmentation_flag], [
                tf.float32, tf.float32, tf.int32, tf.string])))

    return _build_input_aux(dataset, augmentation_flag, buffer_size,
                            batch_size, cfg)


def build_input(data_path, input_list_filepaths, activities_list_path,
                batch_size, supervision_mode, run_mode):
    """
    Builds supervised/unsupervised input tuples for the TwoStream network.

    Args:
        data_path: base-directory containing the dataset.
        input_list_filepaths: list of text files, each listing a set of input
            video files to be used for generating input tuples.
        activities_list_path: path to a text file listing all action-classes,
            so that we can map the class-name to an integer between 0 and
            num-classes - 1.
        batch_size: size for mini-batch.
        supervision_mode (string): either 'supervised' or 'unsupervised'
        run_mode (string): either 'train', 'test' or 'val'

    Returns:
        dataset (tf.data.Dataset): tf dataset of tuples, each tuple consists of
            a center frame, motion-encoding (such as stack-of-difference),
            class-label, and file-name from which this tuple was generated.
    """

    global class_names_all

    cfg = dict()
    cfg['frame_height'] = const.frame_height
    cfg['frame_width'] = const.frame_width
    cfg['context_channels'] = const.context_channels
    buffer_size = const.buffer_size  # TODO: rethink good value for buffer size

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
            dataset = _build_supervised_input(dataset, True, buffer_size,
                                              batch_size, cfg)
        elif run_mode == 'test' or run_mode == 'val':
            dataset = _build_supervised_input(dataset, False, buffer_size,
                                              batch_size, cfg)
        else:
            raise ValueError('Unknow run_mode: %s' % run_mode)
    elif supervision_mode == 'unsupervised':
        if run_mode == 'train':
            dataset = _build_unsupervised_input(dataset, True, buffer_size,
                                                batch_size, cfg)
        elif run_mode == 'test' or run_mode == 'val':
            dataset = _build_unsupervised_input(dataset, False, buffer_size,
                                                batch_size, cfg)
        else:
            raise ValueError('Unknow run_mode: %s' % run_mode)
    else:
        raise ValueError('Unknow supervision_mode: %s' % supervision_mode)

    # dataset = dataset.prefetch(const.buffer_size)

    return dataset
