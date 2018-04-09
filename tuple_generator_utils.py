import constants as const
import cv2
import imageio
import numpy as np
import os.path as osp
import pickle
import traceback


def standardize_frame(frame, crop_coords=None, allow_rotation=False):
    # TODO: I think we should do that, right? Currently it is not activated..
    # if height > width, flip frame 90 degrees
    if allow_rotation and frame.shape[0] > frame.shape[1]:
        frame = np.transpose(frame, [1, 0, 2])

    if crop_coords is not None:
        y_st, x_st = crop_coords
        y_end = min(frame.shape[0], y_st + const.frame_height)
        x_end = min(frame.shape[1], x_st + const.frame_width)
        frame = frame[y_st : y_end, x_st : x_end, :]

    # if frame is still not in standard size, then rescale it
    if(frame.shape[0] != const.frame_height or
       frame.shape[1] != const.frame_width):
        frame = cv2.resize(frame, (const.frame_width, const.frame_height))

    return frame

def get_standard_frame(video, frame_index, crop_coords=None,
                       allow_rotation=False):
    """
    Retreives a specific frame from a video, and rescales it into a pre-defined
    height and width (specified by: const.frame_height and const.frame_width).

    Args:
        video: loaded video with imageio ffmpeg.
        frame_index (int): index of the required frame.
        crop_coords (tupe(int, int)): If not non, then crop the frame starting
            at the given top-left coordinates. If it is None, then just
            rescale the frame (i.e resize it). 
        allow_rotation (bool): whether to rotate the frame 90 degrees to
            maintain that the width is greater than the height.

    Returns:
        array (height x width x 3) representing the extracted frame after being
        rescaled (or cropped) to a standard size.
    """
    frame = video.get_data(frame_index)

    # TODO: I think we should do that, right? Currently it is not activated..
    # if height > width, flip frame 90 degrees
    if allow_rotation and frame.shape[0] > frame.shape[1]:
        frame = np.transpose(frame, [1, 0, 2])

    if crop_coords is not None:
        y_st, x_st = crop_coords
        y_end = min(frame.shape[0], y_st + const.frame_height)
        x_end = min(frame.shape[1], x_st + const.frame_width)
        frame = frame[y_st : y_end, x_st : x_end, :]

    # if frame is still not in standard size, then rescale it
    if(frame.shape[0] != const.frame_height or
       frame.shape[1] != const.frame_width):
        frame = cv2.resize(frame, (const.frame_width, const.frame_height))

    return frame


def create_stack_of_diffs_from_frames(frames, augment_flag):
    """
    Creates a stack of differences by converting the specified frames into
    grayscale and taking their differences.

    Args:
        video: loaded video with imageio ffmpeg.
        frame_inicies (list of ints): frame indices the constitute the stack.
        augment_flag (bool): whether to apply data augmentation on the tuple
            or not. Augmentation can be random cropping, channel splitting,
            ... etc

    Returns:
        array (stack_size x height x width) representing the stack of diffs.
        stack_size is len(frame_indices) - 1
    """
    num_frames = len(frames)
    stack_of_diffs = np.zeros((const.frame_height, const.frame_width,
                               num_frames - 1))

    frame_height, frame_width, _ = frames[0].shape
    if augment_flag:
        rand_rgb_channel = np.random.choice(3)
        rand_crop = np.random.rand();
        crop_y = int(rand_crop * (frame_height - const.frame_height))
        crop_x = int(rand_crop * (frame_width - const.frame_width))
        crop_y = max(0, crop_y)
        crop_x = max(0, crop_x)
        crop_coords = (crop_y, crop_x)
    else:
        crop_coords = None

    current_frame = standardize_frame(frames[0], crop_coords=crop_coords)
    if augment_flag:
        current_frame = np.squeeze(current_frame[:, :, rand_rgb_channel])
    else:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    current_frame = current_frame.astype(np.int32)

    for i in range(num_frames - 1):
        next_frame = standardize_frame(frames[i + 1], crop_coords=crop_coords)
        if augment_flag:
            next_frame = np.squeeze(next_frame[:, :, rand_rgb_channel])
        else:
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        next_frame = next_frame.astype(np.int32)

        stack_of_diffs[:, :, i] = current_frame - next_frame
        current_frame = next_frame

    return stack_of_diffs


def create_stack_of_diffs(video, frame_indices, augment_flag):
    """
    Creates a stack of differences by converting the specified frames into
    grayscale and taking their differences.

    Args:
        video: loaded video with imageio ffmpeg.
        frame_inicies (list of ints): frame indices the constitute the stack.
        augment_flag (bool): whether to apply data augmentation on the tuple
            or not. Augmentation can be random cropping, channel splitting,
            ... etc

    Returns:
        array (stack_size x height x width) representing the stack of diffs.
        stack_size is len(frame_indices) - 1
    """
    num_frames = len(frame_indices)
    stack_of_diffs = np.zeros((const.frame_height, const.frame_width,
                               num_frames - 1))

    frame_height, frame_width = video.get_meta_data()['size']
    if augment_flag:
        rand_rgb_channel = np.random.choice(3)
        rand_crop = np.random.rand();
        crop_y = int(rand_crop * (frame_height - const.frame_height))
        crop_x = int(rand_crop * (frame_width - const.frame_width))
        crop_y = max(0, crop_y)
        crop_x = max(0, crop_x)
        crop_coords = (crop_y, crop_x)
    else:
        crop_coords = None

    current_frame = get_standard_frame(video, frame_indices[0],
                                       crop_coords=crop_coords)
    if augment_flag:
        current_frame = np.squeeze(current_frame[:, :, rand_rgb_channel])
    else:
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    current_frame = current_frame.astype(np.int32)

    for i in range(num_frames - 1):
        next_frame = get_standard_frame(video, frame_indices[i + 1],
                                        crop_coords=crop_coords)
        if augment_flag:
            next_frame = np.squeeze(next_frame[:, :, rand_rgb_channel])
        else:
            next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        next_frame = next_frame.astype(np.int32)

        stack_of_diffs[:, :, i] = current_frame - next_frame
        current_frame = next_frame

    return stack_of_diffs


def split_into_tuples(video_path, chunks, augment_tuples=False):
    """
    Splits a video into consecutive but non-overlapping sequences.

    Args:
        video_path (str): path to source video (e.g. *.avi).
        num_frames (int): size of each sequence.
        step (int): "Default" step between frames within a sampled sequence. It
            can be readjusted for short videos.
        augment_tuples (bool): whether to apply data augmentation on the tuple
            or not. Augmentation can be random cropping, channel splitting,
            ... etc

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

    # Generate center frames and stacks of differences
    center_frames_indices = map(lambda x: (x[-1] + x[0]) // 2, chunks)
    center_frames = list(map(lambda x: get_standard_frame(video, x),
                             center_frames_indices))

    stacks_of_diffs = [None] * len(chunks)
    # FIXME: we need to augment the input either here or by redesign!
    for ii in range(len(chunks)):
        stacks_of_diffs[ii] = create_stack_of_diffs(video, chunks[ii],
                                                    augment_tuples)

    return center_frames, stacks_of_diffs


def visualize_saved_pickle(pkl_file_path, output_dump_path, output_prefix='',
                           verbose=False):
    """
    Visualize center frames and their stack of differences for some video.
    Args:
        pkl_file_path: path the generated pickle file for a video.
        output_dump_path: path to save visualization images.
        output_prefix: prefix to prepend to the names of visualization images.
    """
    with open(pkl_file_path, 'rb') as f:
        # centers, stacks_of_diffs, class_labels, filename = pickle.load(f)
        loaded_pkl = pickle.load(f)

    centers = loaded_pkl[0]
    stacks_of_diffs = loaded_pkl[1]
    class_labels = loaded_pkl[2] if len(loaded_pkl) > 2 else None
    filenames = loaded_pkl[3] if len(loaded_pkl) > 3 else None

    if verbose and class_labels is not None:
        print('class labels = ', class_labels)
    if verbose and filename is not None:
        for fname in filenames:
            print('filename = %s', fname)

    for i in range(len(centers)):
        img = np.uint8(centers[i])
        center_filename = output_prefix + ('_center_%d' % i)
        center_filepath = osp.join(output_dump_path, center_filename)
        # b, g, r = cv2.split(img)
        # img = cv2.merge([r, g, b])
        if verbose:
            print('saving center to %s' % center_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(center_filepath + '.png', img)

        stack = stacks_of_diffs[i]
        stack = (stack + 255) / 2
        for j in range(stack.shape[-1]):
            motion_filename = output_prefix + ('_center_%d_motion_%d' % (i, j))
            motion_filepath = osp.join(output_dump_path, motion_filename)
            if verbose:
                print('saving motion to %s' % motion_filepath)
            img = np.uint8(stack[:, :, j])
            cv2.imwrite(motion_filepath + '.png', img)
