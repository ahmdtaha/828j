import sys
sys.path.append('../')
import time
import utils.confusion_matrix as confusion_matrix
import tensorflow as tf
import os
from nets.two_stream import TwoStreamNet
from nets.motion_tower import  MotionTower
import data_sampling.honda_labels as honda_lbls
import constants as const
import configuration as config
import numpy as np
import utils.os_utils as os_utils
import imageio
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from bisect import bisect
from utils.cache import LRUCache
from utils.metric import compute_average_precision_classification


def load_subset(subset):
    df = pd.read_pickle(config.db_path + '/db_summary_splits.pkl')
    # print(df.shape)
    split = config.db_split

    # The video is included in the training set if id is 1
    # The video is included in the testing set if id is 2
    # The video is not included for training/testing if id is 0
    # print(df.loc[df['video-len'] >= 7])
    if (subset == 'train'):
        sub_df = df.loc[(df['split' + str(split)] == 1) & (df['video-len'] >= 7)]  ##
    elif (subset == 'val'):
        sub_df = df.loc[(df['split' + str(split)] == 0) & (df['video-len'] >= 7)]
    elif (subset == 'test'):
        sub_df = df.loc[(df['split' + str(split)] == 2) & (df['video-len'] >= 7)]

    # print(sub_df.shape)

    videos_names = sub_df['video-name'].tolist()
    videos_lbls = sub_df['video-lbl'].tolist()
    videos_len = sub_df['video-len'].tolist()

    return videos_names, videos_lbls, videos_len

def get_img(frame_idx,session):

    img_path = os.path.join(config.db_path,session,'frame_%04d.jpg' % frame_idx);
    img = imageio.imread(img_path);
    img = cv2.resize(img, (const.frame_height, const.frame_width))
    return img

def imgs2sod(imgs):

    stack_diff = np.zeros((const.frame_height, const.frame_width, const.context_channels))
    frames_order = np.arange(const.context_channels + 1)
    for i in range(const.context_channels):
        current_frame = imgs[frames_order[i]];
        next_frame = imgs[frames_order[i+1]];
        current_frame = cv2.resize(current_frame, (const.frame_height, const.frame_width))
        next_frame = cv2.resize(next_frame, (const.frame_height, const.frame_width))
        stack_diff[:,:,i] = current_frame.astype(np.int32) - next_frame.astype(np.int32);
    return stack_diff;

def get_context(center_idx,session):
    imgs = []
    for idx in [-3,-2,-1,1,2,3]:
        img_path = os.path.join(config.db_path, session, 'frame_%04d.jpg' % (center_idx+idx));
        img = cv2.imread(img_path,0);
        imgs.append(img)
    return imgs2sod(imgs)


if __name__ == '__main__':
    dir = os_utils.get_last_part(config.model_save_path)
    save_result_dir = os.path.join('..', 'dump', dir)
    cnf_mat_path = save_result_dir + '/cnf_mat.pkl'
    print(cnf_mat_path)
    if (not os.path.exists(cnf_mat_path)):

        train_videos, train_videos_lbls, train_videos_len = load_subset('test');
        predictions = np.zeros(len(train_videos_lbls))
        num_labels = config.num_classes;

        isTwoStream = True
        if isTwoStream:
            model = TwoStreamNet(supervised=True, train_spatial_tower=False, train_motion_tower=False,
                                 load_alex_weights=False)
        else:
            model = MotionTower(mode=tf.estimator.ModeKeys.PREDICT)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        model_dir = '../' + config.model_save_path;
        ckpt_file = os.path.join(model_dir, config.model_save_name)
        saver.restore(sess, ckpt_file)
        print('Model loaded successfully')
        cnf_matrix = np.zeros((num_labels,num_labels),dtype=np.int32)
        for idx,vdx in enumerate(train_videos):

            print('Progress : ', idx /len(train_videos))
            num_possible_frames = len(range(3,train_videos_len[idx]-3));
            words = np.zeros((num_possible_frames ,const.frame_height,const.frame_width,const.frame_channels))
            contexts = np.zeros((num_possible_frames, const.frame_height, const.frame_width, const.context_channels))
            i = 0
            for frame in range(3,train_videos_len[idx]-3):
                center_frame = get_img(frame,vdx)
                context = get_context(frame,vdx)
                words[i,:,:,:]  = center_frame
                contexts[i,:,:,:] = context
                i+=1
            if isTwoStream:
                feed_dict = {model.input_words: words,
                             model.input_context: contexts}
            else:
                feed_dict = {model.input_context: contexts}

            actions_prob = sess.run(model.supervised_logits, feed_dict)
            actions_prob = np.sum(actions_prob,axis=0)
            predict = np.argmax(actions_prob);
            cnf_matrix[train_videos_lbls[idx],predict ] +=1
            accuracy = np.trace(cnf_matrix) / np.sum(cnf_matrix)
            print('accuracy ' ,accuracy)
        os_utils.touch_dir(save_result_dir)
        os_utils.pkl_write(save_result_dir + '/cnf_mat.pkl', cnf_matrix)
    else:
        print('Load previously saved files')
        cnf_matrix = os_utils.pkl_read(cnf_mat_path)

    plt.figure()
    confusion_matrix.plot_confusion_matrix(cnf_matrix, classes=[],
                                           title='Confusion Matrix Un-normalization',fmt='.0f')

    plt.show()
    cnf_file = save_result_dir + '/confusion_matrix.png'
    plt.savefig(cnf_file, dpi=300);
    plt.close()

    rows_sum = np.sum(cnf_matrix, axis=1);

    cnf_matrix_percentage = (cnf_matrix / (rows_sum[:, np.newaxis] + 1e-4)) * 100

    confusion_matrix.plot_confusion_matrix(cnf_matrix_percentage, classes=[],
                                           title='Confusion Matrix Percentage', fmt='.0f')

    plt.show()
    cnf_file = save_result_dir + '/norm_confusion_matrix.png'
    plt.savefig(cnf_file, dpi=300);

    accuracy = np.trace(cnf_matrix) / np.sum(cnf_matrix)
    stats = ['Accuracy ' + str(accuracy) + '\n']



    os_utils.txt_write(save_result_dir + '/stats.txt', stats)
