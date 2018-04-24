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
    with open(config.db_path + '/' + subset + '_session.txt', 'r') as fin:
        sessions = fin.read().strip().split('\n')
    num_sessions = len(sessions)
    sessions_annotations = [None] * num_sessions

    for vdz in range(num_sessions):
        vdz_annotation_path = os.path.join(config.db_path, 'labels/' + sessions[vdz] + '_goal.pkl')
        sessions_annotations[vdz] = os_utils.pkl_read(vdz_annotation_path)

    return sessions, num_sessions, sessions_annotations

def imgs2sod(imgs):

    stack_diff = np.zeros((const.frame_height, const.frame_width, const.context_channels),dtype=np.int32)

    for i in range(const.context_channels):
        current_frame = imgs[i,:, :];
        next_frame = imgs[i+1,:,:];
        stack_diff[:,:,i] = current_frame - next_frame;

    return stack_diff;

def read_rgb_img(img_path):
    result = img_caches.get(img_path)
    if result is None:
        img = imageio.imread(img_path);
        img = cv2.resize(img, (const.frame_height, const.frame_width))
        img_caches.set(img_path, img)
        return img
    else:
        return result

def read_image_from_path(img_path):
    result =  sod_caches.get(img_path)
    if result is None:
        img = read_rgb_img(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.int32);
        sod_caches.set(img_path,img)
        return img
    else:
        return result

def get_context(center_idx,session):
    imgs = np.zeros((const.context_channels+1,const.frame_height,const.frame_width))
    index = 0;
    for idx in [-5,-3,-1,1,3,5]:
        img_path = os.path.join(config.db_path, 'frames', session, 'frame_%04d.jpg' % (center_idx+idx));
        if (not os.path.exists(img_path)):
            img_path = os.path.join(config.db_path, 'frames', session,
                                    'frame_%05d.jpg' % (center_idx + idx));
        img=read_image_from_path(img_path)
        imgs[index,:,:] = img;
        index +=1
        #imgs.append(img)
    return imgs2sod(imgs)

def get_img(frame_idx,session):

    img_path = os.path.join(config.db_path,'frames',session,'frame_%04d.jpg' % frame_idx);
    if(not os.path.exists(img_path )):
        img_path = os.path.join(config.db_path, 'frames', session, 'frame_%05d.jpg' % frame_idx);
    # img = imageio.imread(img_path);
    # img = cv2.resize(img , (const.frame_height, const.frame_width))
    img = read_rgb_img(img_path);
    return img

def eval_by_frame():
    cnf_matrix = np.zeros((config.num_classes, config.num_classes), dtype=np.int32)
    event_lst, gt_label_lst, pred_label_lst, score_lst = [], [], [], []
    event_id = 0
    batch_size = 100;
    words = np.zeros((batch_size ,const.frame_height,const.frame_width,const.frame_channels))
    contexts = np.zeros((batch_size , const.frame_height, const.frame_width, const.context_channels))


    for session in range(num_test_sessions):
        num_frames = test_sessions_annotations[session]['s'][-1] -5;
        #print(test_sessions_annotations[session]['s'])
        #for center_frame_idx in range(6,num_frames-5): ## Start at 6 because frames start at 1!!
        print('Number of frames' ,num_frames ,' in ',test_sessions[session]);
        #num_frames = 97
        for center_frame_idx in range(6, num_frames ,batch_size ):  ## Start at 6 because frames start at 1!!

            max_batch_frame = min(center_frame_idx+batch_size ,num_frames)
            num_frame_in_batch = max_batch_frame - center_frame_idx;
            print('From ', center_frame_idx,' to ',max_batch_frame );
            start_time = time.time()
            for idx  in range(num_frame_in_batch):
                #word_start_time = time.time()
                word = get_img(idx + center_frame_idx  , test_sessions[session])
                #print(idx + center_frame_idx)
                # elapsed_time = time.time() - word_start_time
                # print('word :', elapsed_time)
                # word_start_time = time.time()
                context = get_context(idx + center_frame_idx, test_sessions[session])
                # elapsed_time = time.time() - word_start_time
                # print('context :', elapsed_time)
                # word_start_time = time.time()
                words[idx,:,:,:] = word
                contexts[idx, :, :, :] = context
                # elapsed_time = time.time() - word_start_time
                # print('fit in numpy :', elapsed_time)

            elapsed_time = time.time() - start_time
            # print('Memory :', elapsed_time)
            if isTwoStream:
                feed_dict = {model.input_words: words,
                             model.input_context: contexts}
            else:
                feed_dict = {model.input_context: contexts}
            start_time = time.time()
            actions_prob = sess.run(model.supervised_logits, feed_dict)
            elapsed_time = time.time() - start_time
            # print('GPU :', elapsed_time)
            probability_predictions = np.argmax(actions_prob, axis=1);
            for idx in range(num_frame_in_batch):
                goal_idx = bisect(test_sessions_annotations[session]['s'],idx+ center_frame_idx )
                event_goal = test_sessions_annotations[session]['G'][goal_idx-1];

                event_lst.append(event_id)
                gt_label_lst.append(event_goal)
                predicted_label = probability_predictions[idx];
                pred_label_lst.append(predicted_label )
                score_lst.append(actions_prob[idx, predicted_label ])
                event_id += 1
                cnf_matrix[event_goal, predicted_label ] += 1

    ground_truth = pd.DataFrame({'event-id': event_lst,
                                 'label': gt_label_lst})
    ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)

    prediction = pd.DataFrame({'event-id': event_lst,
                               'label': pred_label_lst,
                               'score': score_lst})

    return cnf_matrix, ground_truth, prediction

def eval_by_event():
    cnf_matrix = np.zeros((config.num_classes, config.num_classes), dtype=np.int32)
    event_lst, gt_label_lst, pred_label_lst, score_lst = [], [], [], []
    event_id = 0
    for session in range(num_test_sessions):
        for goal_idx, event_goal in enumerate(test_sessions_annotations[session]['G']):
            event_start = test_sessions_annotations[session]['s'][goal_idx]
            event_end = test_sessions_annotations[session]['s'][goal_idx + 1]
            # print(event_start,event_end,honda_lbls.honda_num2labels[event_goal])
            center_idx = (event_start + event_end) // 2
            if event_end - event_start > 5 and center_idx > 5 and \
                            center_idx < test_sessions_annotations[session]['s'][-1] - 6:
                ## valid event
                word = get_img(center_idx, test_sessions[session])
                context = get_context(center_idx, test_sessions[session])
                word = word[np.newaxis, :, :, :]
                context = context[np.newaxis, :, :, :]

                if isTwoStream:
                    feed_dict = {model.input_words: word,
                                 model.input_context: context}
                else:
                    feed_dict = {model.input_context: context}
                actions_prob = sess.run(model.supervised_logits, feed_dict)
                # print(np.argmax(actions_prob),event_goal)

                event_lst.append(event_id)
                gt_label_lst.append(event_goal)
                max_prediction = np.argmax(actions_prob);
                pred_label_lst.append(max_prediction)
                score_lst.append(actions_prob[0, max_prediction])
                event_id += 1

                cnf_matrix[event_goal, np.argmax(actions_prob)] += 1
                # print(cnf_matrix)
                # print(np.round(cnf_matrix/ (np.sum(cnf_matrix, axis=1)+ 1e-4),2))

    ground_truth = pd.DataFrame({'event-id': event_lst,
                                 'label': gt_label_lst})
    ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)

    prediction = pd.DataFrame({'event-id': event_lst,
                               'label': pred_label_lst,
                               'score': score_lst})

    return cnf_matrix,ground_truth,prediction
if __name__ == '__main__':
    isTwoStream = config.use_two_stream
    img_caches = LRUCache(150);
    sod_caches = LRUCache(150);
    if isTwoStream:
        model = TwoStreamNet(supervised=True,train_spatial_tower=False,train_motion_tower=False,load_alex_weights=False)
    else:
        model = MotionTower(mode=tf.estimator.ModeKeys.PREDICT)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    dir = os_utils.get_last_part(config.model_save_path)
    save_result_dir = os.path.join('..','dump',dir)
    model_dir = '../'+config.model_save_path;
    ckpt_file = os.path.join(model_dir , config.model_save_name)
    saver.restore(sess, ckpt_file)
    print('Model loaded successfully')


    cnf_mat_path = save_result_dir+'/cnf_mat.pkl'
    print(cnf_mat_path )
    if(not os.path.exists(cnf_mat_path)):
        print('CNN prediction')
        ## Prepare data tuples
        test_sessions, num_test_sessions, test_sessions_annotations = load_subset('test');
        print(test_sessions)
        print('Number of sessions : ',num_test_sessions)

        cnf_matrix, ground_truth, prediction = eval_by_frame();

        os_utils.touch_dir(save_result_dir)
        os_utils.pkl_write(save_result_dir + '/cnf_mat.pkl', cnf_matrix)
        ground_truth.to_pickle(save_result_dir + '/ground_truth.pkl')
        prediction.to_pickle(save_result_dir + '/prediction.pkl')
    else:
        print('Load previously saved files')
        cnf_matrix = os_utils.pkl_read(cnf_mat_path)
        ground_truth = pd.read_pickle(save_result_dir + '/ground_truth.pkl')
        prediction = pd.read_pickle(save_result_dir + '/prediction.pkl')



    confusion_matrix_labels = ["Background", "Intersection", "L turn", 'R turn',
                               'L ln change', 'R ln change', 'Crosswalk',
                               'U-turn', 'L ln branch', 'R ln branch', 'Merge']


    # Plot non-normalized confusion matrix
    plt.figure()
    confusion_matrix.plot_confusion_matrix(cnf_matrix, classes=confusion_matrix_labels,
                          title='Confusion Matrix Un-normalization')

    plt.show()
    cnf_file = save_result_dir +'/confusion_matrix.png'
    plt.savefig(cnf_file, dpi=300);
    plt.close()

    rows_sum = np.sum(cnf_matrix, axis=1);

    cnf_matrix_percentage = (cnf_matrix / (rows_sum [:,np.newaxis] + 1e-4)) * 100

    confusion_matrix.plot_confusion_matrix(cnf_matrix_percentage, classes=confusion_matrix_labels,
                                           title='Confusion Matrix Percentage',fmt='.0f')

    plt.show()
    cnf_file = save_result_dir +'/norm_confusion_matrix.png'
    plt.savefig(cnf_file, dpi=300);

    accuracy= np.trace(cnf_matrix) / np.sum(cnf_matrix)
    stats = ['Accuracy '+ str(accuracy) + '\n']

    ap = np.zeros(config.num_classes)
    for event in range(config.num_classes):
        gt_idx = ground_truth['label'] == event
        pred_idx = prediction['label'] == event
        ap[event] = compute_average_precision_classification(
            ground_truth.loc[gt_idx].reset_index(drop=True),
            prediction.loc[pred_idx].reset_index(drop=True))
        #print(event, honda_lbls.honda_num2labels[event], ap[event])
        stats.append(str(event)+ '\t' +  honda_lbls.honda_num2labels[event]+'\t'+ str(ap[event]))

    print('MAP with background', np.mean(ap))
    print('MAP with background', np.mean(ap[1:]))
    stats.append('mAp with Background '+ str(np.mean(ap)))
    stats.append('mAp without Background ' + str(np.mean(ap[1:])))

    os_utils.txt_write(save_result_dir +'/stats.txt',stats )




