import os.path as osp
import pickle
import sys
sys.path.append('../')
import tensorflow as tf
import tf_tuple_generator as tuple_gen
import tuple_generator_utils as gen_utils


# TODO: possibly move some of these consts to configuration.py
# Constants
k_base_dir = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning'
k_dataset_path = k_base_dir + '/datasets/ucf101'
k_input_list_filepath = k_base_dir + '/datasets/ucfTrainTestlist/tmp_list.txt'
k_activities_path = k_base_dir + '/datasets/ucfTrainTestlist/activities'
k_batch_size = 4
k_supervision_mode = 'supervised'
k_run_mode = 'train'
k_log_root = k_base_dir + '/tf_logs'
k_output_dump_path = k_base_dir + '/outputs'


def test_build_supervised_input_for_train():
    center_frames_op, motion_encoding_op, class_label_op, filenames_op = \
        tuple_gen.build_input(k_dataset_path, k_input_list_filepath,
                              k_activities_path, k_batch_size,
                              k_supervision_mode, k_run_mode, k_log_root,
                              num_threads=16)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tf.train.start_queue_runners(sess)

    num_batches = 4
    for batch_i in range(num_batches):
        (center_frames, motion_encodings, class_labels, filenames) = sess.run(
            [center_frames_op, motion_encoding_op, class_label_op,
             filenames_op])

        print('Storing sample output in ' + k_output_dump_path)
        for i in xrange(k_batch_size):
            print('Storing output for batch #%d, sample #%d (total: %d)' % (
                  batch_i, i, batch_i*k_batch_size + i + 1))
            print('class_label = %d -- filename = %s' % (class_labels[i],
                                                         filenames[i]))
            if k_batch_size > 1:
                basename, ext = osp.splitext(osp.basename(filenames[i]))
            else:
                basename, ext = osp.splitext(osp.basename(filenames))

            # save/pickle tuples
            out_pkl_path = osp.join(k_output_dump_path, basename + '.pkl') 
            with open(out_pkl_path, 'wb') as f:
                pickle.dump((center_frames, motion_encodings, class_labels,
                             filenames), f)

            # visualize tuples
            gen_utils.visualize_saved_pickle(
                out_pkl_path, k_output_dump_path,
                output_prefix='test_sup_train_tuples_')


def test_split_into_train_tuples():
    video_path = k_dataset_path + '/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'

    with open(k_activities_path, 'r') as f:
        tuple_gen.class_names_all = [x.strip() for x in f.readlines()]

    print('Splitting video %s' % video_path)
    centers, motion_encs, class_labels, paths_tiled = \
        tuple_gen._split_into_train_tuples(video_path)

    print('class_labels = ', class_labels)
    for file_path in paths_tiled:
        print('file path = %s' % file_path)

    # save/pickle tuples
    out_pkl_path = osp.join(k_output_dump_path,
                            'test_split_into_train_tuples.pkl')
    with open(out_pkl_path, 'wb') as f:
        pickle.dump((centers, motion_encs, class_labels, paths_tiled), f)

    # visualize tuples
    gen_utils.visualize_saved_pickle(out_pkl_path, k_output_dump_path,
                                    'test_split_into_train_tuples_')


if __name__ == '__main__':
    test_build_supervised_input_for_train()
    # test_split_into_train_tuples()
