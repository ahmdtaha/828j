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
import sys
import time
import traceback
import tuple_generation_utils as gen_utils


def _generate_and_save_test_tuples(video_path, output_path, sampling_fn):
    """
    Generates and saves test tuple(s) for one video, and pickles the output in
    the format (center_frames, stacks_of_diffs).

    Args:
        video_path (str): path to input video.
        output_path (str): output path to save all tuples.
        sampling_fn (function pointer): pointer to the function that takes in
            a video and samples input sequences from it (e.g. random sampling,
            consecutive sampling or, constrained consecutive sampling)
    """
    center_frames, stacks_of_diffs = sampling_fn(video_path)
    # print('Done with %s: generated %d test tuples' % (video_path,
    #                                                   len(center_frames)))
    with open(output_path, 'wb') as f:
        pickle.dump((center_frames, stacks_of_diffs), f)


# FIXME: something is worng: several videos of hmdb51 are skipped (they crash
# but don't print any error messages when working in multiple threads!)
# Failures are not entirely deterministic!!
def _split_into_test_tuples(video_path, num_frames=6, step=15):
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

    # Non-overlapping chunks: e.g. if step = 15, then:
    # [1,16,31,46,61,76]--[77,92,107,122,137,152]--[153,...]--...
    chunks = []
    start_frame = 0
    while start_frame <= total_num_frames:
        chunk = np.arange(
            start_frame, start_frame + (num_frames - 1) * step + 1, step)
        if chunk[-1] < total_num_frames:
            chunks.append(chunk)
            start_frame = chunk[-1] + 1
        else:
            # if remaining frames cannot complete a chunk, then center previous
            # chunks so that leftovers are equally two-sided
            shift = (total_num_frames - start_frame + 1) // 2
            for i in range(len(chunks)):
                chunks[i] = chunks[i] + shift
            break

    center_frames, stacks_of_diffs = gen_utils.split_into_tuples(video_path,
                                                                 chunks)

    return center_frames, stacks_of_diffs


def _process_split_list(parent_dir, video_list, save_dir, num_threads):
    """
    Generates and saves test tuples for a list of videos. It operates in
    multi-threads for efficiency.

    Args:
        parent_dir (str): parent directory containing the input list of videos.
        video_list (list): list of strings of video names to test. For example:
                           ['vid_1.avi', 'vid_2.avi', ...]
        save_dir (str): directory to save generated test tuples (pickle files).
        num_threads (int): number of threads for parallel processing.
    """
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    pool = mp.Pool(num_threads)

    for video_name in video_list:
        video_path = osp.join(parent_dir, video_name)
        basename, _ = osp.splitext(video_name)
        tuples_save_path = osp.join(save_dir, basename + "-test.pkl")

        # Skip if already exists
        if osp.exists(tuples_save_path):
            print('Skipping file %s: already processed' % tuples_save_path)
            continue

        # Divide into 6-frame tuples
        pool.apply_async(_generate_and_save_test_tuples, args=(
            video_path, tuples_save_path, _split_into_test_tuples))
        # _generate_and_save_test_tuples(
        #     video_path, tuples_save_path, _split_into_test_tuples)

    pool.close()
    pool.join()


def _process_hmdb51(dataset_path, splits_dir, test_tuples_basedir,
                    num_threads=16):
    """
    Generates and saves test tuples for all test videos of the HMDB51 dataset.
    It operates in multi-threads for efficiency.

    Args:
        dataset_path (str): path to the HMDB51 dataset.
        splits_dir (str): path to directory with text files describing how the
            dataset is split (3 different splits and train-val-test)
        test_tuples_basedir (str): directory to save generated test tuples.
        num_threads (int): number of threads for parallel processing.
    """
    num_processed_files = 0
    # Loop over each split file (dataset is divided into three splits, each
    # split is then divided into val-train-test)
    for f_name in os.listdir(splits_dir):
        # f_name is in the format <activity_name>_test_split<splist_id>.txt
        activity = f_name[:-16]
        split_id = ord(f_name[-5]) - ord('0')
        activity_dir = osp.join(dataset_path, activity)
        activity_out_dir = osp.join(test_tuples_basedir, activity)
        save_dir = activity_out_dir + '_test_tuples_split%d' % split_id
        with open(osp.join(splits_dir, f_name), 'r') as f:
            lines = [x.strip() for x in f.readlines()]
        test_lines = filter(lambda line: line[-1] == '2', lines)
        test_videos = list(map(lambda line: str.split(line)[0], test_lines))

        _process_split_list(activity_dir, test_videos, save_dir, num_threads)

        # Print progress info
        num_processed_files += len(test_videos)
        print('Processed file %s with %d test videos' % (f_name,
                                                         len(test_videos)))
        print('Total processed files so far = %d' % num_processed_files)


def _process_ucf101(dataset_path, splits_dir, test_tuples_basedir,
                    num_threads=16):
    """
    Generates and saves test tuples for all test videos of the UCF101 dataset.
    It operates in multi-threads for efficiency.

    Args:
        dataset_path (str): path to the HMDB51 dataset.
        splits_dir (str): path to directory with text files describing how the
            dataset is split (3 different splits and train-val-test)
        test_tuples_basedir (str): directory to save generated test tuples.
        num_threads (int): number of threads for parallel processing.
    """
    num_processed_files = 0

    # Loop over each split file (dataset is divided into three splits, each
    # split is then divided into val-train-test)
    for split_id in range(1, 4):
        f_name = 'testlist%02d.txt' % split_id
        with open(osp.join(splits_dir, f_name), 'r') as f:
            lines = [x.strip() for x in f.readlines()]

        parsed_lines = np.asarray(list(map(lambda x: x.split('/'), lines)))
        videos_label_name = parsed_lines[:, 0]
        videos_name = parsed_lines[:, 1]
        action_classes = np.unique(videos_label_name)
        assert len(action_classes) == 101, 'Error parsing the UCF101 dataset'

        for activity in action_classes:
            activity_dir = osp.join(dataset_path, activity)
            activity_out_dir = osp.join(test_tuples_basedir, activity)
            save_dir = activity_out_dir + '_test_tuples_split%d' % split_id
            activity_videos_indices = list(filter(
                lambda index: videos_label_name[index] == activity,
                range(len(videos_name))))
            activity_videos = videos_name[activity_videos_indices]

            _process_split_list(activity_dir, activity_videos, save_dir,
                                num_threads)
            # Print progress info
            num_processed_files += len(activity_videos)
            print('Processed activity %s-split%d with %d test videos' % (
                activity, split_id, len(activity_videos)))
            print('Total processed files so far = %d' % num_processed_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', type=int, default=32)
    parser.add_argument('--dataset_name', help='either: hmdb51 or ucf101',
                        type=str, default='ucf101')
    parser.add_argument('--dataset_path', type=str, default='Datasets/ucf101')
    parser.add_argument(
        '--split_metadata_dir', help='path to where train-eval-test split '
        'info of the dataset is stored', type=str,
        default='Datasets/ucf101/ucfTrainTestlist')
    parser.add_argument(
        '--test_tuples_basedir', help='parent directory to where generated '
        'test tuples are stored', type=str, default='Datasets/ucf101')

    args = parser.parse_args()
    if args.dataset_path[-1] == '/':
        args.dataset_path = args.dataset_path[:-1]

    start_time = time.time()
    if args.dataset_name == 'hmdb51':
        _process_hmdb51(args.dataset_path, args.split_metadata_dir,
                        args.test_tuples_basedir, args.num_threads)
    elif args.dataset_name == 'ucf101':
        _process_ucf101(args.dataset_path, args.split_metadata_dir,
                        args.test_tuples_basedir, args.num_threads)

    runtime = time.time() - start_time
    print('runtime = ', runtime)
