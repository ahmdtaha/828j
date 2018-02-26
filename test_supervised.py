from nets.two_stream import TwoStreamNet
import argparse
import numpy as np
import os
import os.path as osp
import pickle
import tensorflow as tf


def _compute_best_action(video_splits_scores, method='max_log_likelihood'):
    """
    Compares the classification scores for all action classes and chooses the
    top-scoring class as the action label.

    Args:
        video_splits_scores: list of arrays, each list item corresponds to the
            network outptut for a sampled tuple/split/sequence, represented by
            a 1 x num_classes array
        method (string): comparison protocol: either max_log_likelihood or
                                              average_predictions.

    Returns (int):
       predicted action label
    """
    num_action_classes = len(video_splits_scores[-1])
    aggregate_scores = np.zeros(num_action_classes)
    if method == 'max_log_likelihood':
        for split_scores in video_splits_scores:
            aggregate_scores += split_scores
    elif method == 'average_predictions':
        # TODO: CORRECTLY implement average_predictions criterion.
        for split_scores in video_splits_scores:
            prediction = np.argmax(split_scores)
            aggregate_scores[prediction] += 1
    else:
        raise Exception('Error: Unsupported evaluation criterion!')

    return np.argmax(aggregate_scores)


def perform_action_recognition(model, video_pkl_path, sess):
    """
    Classify the video into one of the action classes.

    Args:
        model (TF model): tensorflow trained model.
        video_pkl_path (str): path to pre-generated test tuples for a test
                              video (as a pickle file).
        sess (TF session): tensorflow session to run within.

    Returns (int):
       predicted action label
    """
    with open(video_pkl_path, 'rb') as f:
        centers, stacks_of_diffs = pickle.load(f)
    video_splits_scores = []
    for ii in range(len(centers)):
        center = np.expand_dims(centers[ii], axis=0)
        stack_of_diffs = np.expand_dims(stacks_of_diffs[ii], axis=0)
        feed_dict = {model.input_words: center,
                     model.input_context: stack_of_diffs}
        actions_prob = sess.run(model.supervised_logits, feed_dict)
        actions_prob = np.squeeze(np.asarray(actions_prob))
        video_splits_scores.append(actions_prob)

    lbl = _compute_best_action(video_splits_scores,
                               method='max_log_likelihood')
    return lbl


def _test_videos(model, test_tuples_dir, video_list, sess):
    """
    Classify a list of videos into their predicted action classes.

    Args:
        model (TF model): tensorflow trained model.
        test_tuples_dir (str): parent directory where pre-generated test tuples
                               are stored.
        video_list (list): list of strings of video names to test. For example:
                           ['vid_1.avi', 'vid_2.avi', ...]
        sess (TF session): tensorflow session to run within.

    Returns (1D array):
       predicted action labels for the input list of videos
    """
# TODO: should we handle corrupted files here?
    # video_list_all = video_list
    # video_list = [x for x in video_list if osp.exists(osp.join(
    #     test_tuples_dir, osp.splitext(x)[0] + '-test.pkl'))]
    # if len(video_list) != len(video_list_all):
    #     print('WARNING: skipping %d corrupted test videos in %s!' % (
    #         (len(video_list_all) - len(video_list)), test_tuples_dir))
    test_labels = np.zeros(len(video_list))
    # for video_name in video_list:
    for ii in range(len(video_list)):
        video_name = video_list[ii]
        basename, _ = osp.splitext(video_name)
        video_pkl_name = basename + '-test.pkl'
        video_pkl_path = osp.join(test_tuples_dir, video_pkl_name)

        # Divide into 6-frame tuples
        test_labels[ii] = perform_action_recognition(
            model, video_pkl_path, sess)

    return test_labels


def compute_classification_accuracy(num_splits, num_classes, activities,
                                    predictions_all, dataset_name,
                                    target_split_id=None):
    """
    Computes the mean classification accuracy for a dataset

    Args:
        num_splits (int): number of different splits of a dataset (e.g. in
            HMDB51 and UCF101, there are 3 splits)
        num_classes (int): number of action classes in the dataset.
        activities (list): list of action classes in the dataset
        predictions_all (2D array): (num_splits x num_classes) array of lists,
            each item is a list of predictions for the corresponding action
            class and split id.
        dataset_name  (str): name of the dataset
        target_split_id (int): [OPTIONAL] specifies a specific split for which
            to compute the accuracy.
    """
    splits_accuracy = np.zeros(num_splits)
    for split_id in range(1, 4):
        # if testing only a specific split, ignore files from other splits
        classes_correct_cnt = np.zeros(num_classes)
        classes_total = np.zeros(num_classes)
        if target_split_id is not None and split_id != target_split_id:
            continue

        print('Results for %s, split #%id:' % (dataset_name, split_id))
        for activity_id in range(num_classes):
            correct_cnt = np.sum(
                predictions_all[split_id - 1][activity_id] == activity_id)
            total = len(predictions_all[split_id - 1][activity_id])
            classes_correct_cnt[activity_id] = correct_cnt
            classes_total[activity_id] = total
            acc = correct_cnt * 100.0 / total
            print('%02d: %s: %.2f%%' % (
                activity_id, activities[activity_id], acc))

        split_acc = np.sum(classes_correct_cnt) * 100.0 / np.sum(classes_total)
        print('Mean accuracy for %s-split %d = %.2f%%' % (
            dataset_name, split_id, split_acc))
        print('------------------------------------------------\n')
        splits_accuracy[split_id - 1] = split_acc

    print('%s Summary Results for all 3 splits: ' % dataset_name,
          splits_accuracy)
    if target_split_id is not None:
        print('Mean accuracy over 3 splits = %.2f' % np.average(
              splits_accuracy))


def test_hmdb51(dataset_path, model_ckpt_file, test_tuples_parent_dir,
                split_metadata_dir, target_split_id=None):
    """
    Computes and prints the average classification accuracy for the HMDB51
    dataset.

    Args:
        dataset_path (str): path to the HMDB51 dataset.
        model_ckpt_file (str): path the trained model.
        test_tuples_parent_dir (str): parent directory where pre-generated test
                                      tuples are stored.
        split_metadata_dir (str): path to directory with text files describing
            how the dataset is split (3 different splits and train-val-test)
        target_split_id (int) - optional: pass if you want to evaluate a
            specific split only.
    """
    k_num_splits = 3
    k_num_classes = 51

    # Load trained model
    print('Loading model for the %s dataset from %s' % (
        dataset_path, model_ckpt_file))
    if not osp.exists(model_ckpt_file):
        raise Exception("Error: Couldn't find model at %s!" % model_ckpt_file)

    sess = tf.InteractiveSession()
# FIXME: 1) ask taha about the paramter 'train_alexnet'
# FIXME: 2) how to construct the corresponding arch to the target dataset?!
# it should be passed as a parameter, not coded in file constans.num_classes
# FIXME: 3) batch size needs to be passed as a parameter! (@test sz=1)
    model = TwoStreamNet(supervised=True, train_alexnet=False)
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(model_arch_path)
    saver.restore(sess, model_ckpt_file)
    print('Model loaded successfully')

    # test_labels is 3 x 51 2D array of lists (3 splits x 51 activities)
    test_labels_all = np.reshape([None] * k_num_classes * k_num_splits, (
        k_num_splits, k_num_classes))

    with open(osp.join(dataset_path, 'activities_list'), 'r') as f:
        activities = [x.strip() for x in f.readlines()]
    assert len(activities) == k_num_classes, 'Error reading HMDB51 classes!'

    # Loop over all split files (3 split files per action) of HMDB51
    for f_name in os.listdir(split_metadata_dir):
        # f_name is in the format <activity_name>_test_split<splist_id>.txt
        activity = f_name[:-16]
        activity_id = activities.index(activity)
        split_id = ord(f_name[-5]) - ord('0')

        # if testing only a specific split, ignore files from other splits
        if target_split_id is not None and split_id != target_split_id:
            continue

        activity_test_dir_name = activity + '_test_tuples_split%d' % split_id
        activity_test_dir = osp.join(test_tuples_parent_dir,
                                     activity_test_dir_name)
        with open(osp.join(split_metadata_dir, f_name), 'r') as f:
            lines = [x.strip() for x in f.readlines()]
        # Filter out train and validation videos
        test_lines = filter(lambda line: line[-1] == '2', lines)
        test_videos = list(map(lambda line: str.split(line)[0], test_lines))

        lbls = _test_videos(model, activity_test_dir, test_videos, sess)
        test_labels_all[split_id - 1][activity_id] = lbls

    # Compute accuracy metric
    compute_classification_accuracy(k_num_splits, k_num_classes,
                                    activities, test_labels_all, 'HMDB51',
                                    target_split_id)


def test_ucf101(dataset_path, model_ckpt_file, test_tuples_parent_dir,
                split_metadata_dir, target_split_id=None):
    """
    Computes and prints the average classification accuracy for the HMDB51
    dataset.

    Args:
        dataset_path (str): path to the UCF101 dataset.
        model_ckpt_file (str): path the trained model.
        test_tuples_parent_dir (str): parent directory where pre-generated test
                                      tuples are stored.
        split_metadata_dir (str): path to directory with text files describing
            how the dataset is split (3 different splits and train-val-test)
        target_split_id (int) - optional: pass if you want to evaluate a
            specific split only.
    """
    k_num_splits = 3
    k_num_classes = 101

    # Load trained model
    print('Loading model for the %s dataset from %s' % (
        dataset_path, model_ckpt_file))
    if not osp.exists(model_ckpt_file):
        raise Exception("Error: Couldn't find model at %s!" % model_ckpt_file)

    sess = tf.InteractiveSession()
# FIXME: 2) how to construct the corresponding arch to the target dataset?!
# it should be passed as a parameter, not coded in file constans.num_classes
# FIXME: 3) batch size needs to be passed as a parameter! (@test sz=1)
    model = TwoStreamNet(supervised=True, train_alexnet=False)
    # sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(model_arch_path)
    saver.restore(sess, model_ckpt_file)
    print('Model loaded successfully')

    # test_labels is 3 x 101 2D array of lists (3 splits x 51 activities)
    test_labels_all = np.reshape([None] * k_num_classes * k_num_splits, (
        k_num_splits, k_num_classes))

    with open(osp.join(dataset_path, 'activities_list'), 'r') as f:
        activities = [x.strip() for x in f.readlines()]
    assert len(activities) == k_num_classes, 'Error reading HMDB51 classes!'

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
            activity_id = activities.index(activity)
            activity_test_dir_name = (
                activity + '_test_tuples_split%d' % split_id)
            activity_test_dir = osp.join(test_tuples_parent_dir,
                                         activity_test_dir_name)
            activity_videos_indices = list(filter(
                lambda index: videos_label_name[index] == activity,
                range(len(videos_name))))
            activity_videos = videos_name[activity_videos_indices]

            lbls = _test_videos(model, activity_test_dir, test_videos, sess)
            test_labels_all[split_id - 1][activity_id] = lbls

    # Compute accuracy metric
    compute_classification_accuracy(k_num_splits, k_num_classes,
                                    activities, test_labels_all, 'UCF101',
                                    target_split_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_name', help='dataset name: hmdb51 or ucf101',
        type=str, default='ucf101')
    parser.add_argument(
        '--model_save_dir', help='Path to the *.ckpt model file',
        type=str, default='data/models/hmdb51.cpkt')
    parser.add_argument(
        '--dataset_path', type=str, default='datasets/ucf101')
    parser.add_argument(
        '--test_tuples_basedir', help='parent directory to where generated '
        'test tuples are stored', type=str, default='datasets/ucf101')
    parser.add_argument(
        '--split_metadata_dir', help='path to where train-eval-test split '
        'info of the dataset is stored', type=str,
        default='datasets/ucf101/ucfTrainTestlist')

    args = parser.parse_args()

    if args.dataset_name == 'hmdb51':
        test_hmdb51(args.dataset_path, args.model_save_dir,
                    args.test_tuples_basedir, args.split_metadata_dir)
    elif args.dataset_name == 'ucf101':
        test_ucf101(args.dataset_path, args.model_save_dir,
                    args.test_tuples_basedir, args.split_metadata_dir)
