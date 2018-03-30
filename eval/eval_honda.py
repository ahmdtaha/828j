import sys
sys.path.append('../')
import utils.confusion_matrix as confusion_matrix
import tensorflow as tf
import os
from nets.two_stream import TwoStreamNet
import constants as const
import configuration as config
import numpy as np
import utils.os_utils as os_utils
import imageio
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_subset(subset):
    with open(config.honda_session_path + '/' + subset + '_session.txt', 'r') as fin:
        sessions = fin.read().strip().split('\n')
    num_sessions = len(sessions)
    sessions_annotations = [None] * num_sessions

    for vdz in range(num_sessions):
        vdz_annotation_path = os.path.join(config.honda_session_path, 'labels/' + sessions[vdz] + '_goal.pkl')
        sessions_annotations[vdz] = os_utils.pkl_read(vdz_annotation_path)

    return sessions, num_sessions, sessions_annotations

def imgs2sod(imgs):

    stack_diff = np.zeros((const.frame_height, const.frame_width, const.context_channels))

    for i in range(const.context_channels):
        current_frame = imgs[i][:, :];
        next_frame = imgs[i+1][:,:];

        current_frame = cv2.resize(current_frame, (const.frame_height, const.frame_width))
        next_frame = cv2.resize(next_frame, (const.frame_height, const.frame_width))

        stack_diff[:,:,i] = current_frame.astype(np.int32) - next_frame.astype(np.int32);

    return stack_diff;
def get_context(center_idx,session):
    imgs = []
    for idx in [-5,-3,-1,1,3,5]:
        img_path = os.path.join(config.honda_session_path, 'frames', session, 'frame_%04d.jpg' % (center_idx+idx));
        if (not os.path.exists(img_path)):
            img_path = os.path.join(config.honda_session_path, 'frames', session,
                                    'frame_%05d.jpg' % (center_idx + idx));
        img = imageio.imread(img_path);
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgs.append(img)
    return imgs2sod(imgs)

def get_img(frame_idx,session):

    img_path = os.path.join(config.honda_session_path,'frames',session,'frame_%04d.jpg' % frame_idx);
    if(not os.path.exists(img_path )):
        img_path = os.path.join(config.honda_session_path, 'frames', session, 'frame_%05d.jpg' % frame_idx);
    img = imageio.imread(img_path);
    img = cv2.resize(img , (const.frame_height, const.frame_width))
    return img

if __name__ == '__main__':
    model = TwoStreamNet(supervised=True,train_spatial_tower=False,train_motion_tower=False,load_alex_weights=False)
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
        ## Prepare data tuples
        test_sessions, num_test_sessions, test_sessions_annotations = load_subset('val');
        print(test_sessions)
        print(num_test_sessions)
        cnf_matrix = np.zeros((config.num_classes,config.num_classes),dtype=np.int32)
        for session in range(num_test_sessions):
            for goal_idx, event_goal in enumerate(test_sessions_annotations[session]['G']):
                event_start = test_sessions_annotations[session]['s'][goal_idx]
                event_end = test_sessions_annotations[session]['s'][goal_idx+1]
                #print(event_start,event_end,honda_lbls.honda_num2labels[event_goal])
                center_idx = (event_start + event_end) // 2
                if event_end - event_start > 5 and center_idx > 5 and \
                                center_idx < test_sessions_annotations[session]['s'][-1] - 6:
                    ## valid event
                    word = get_img(center_idx,test_sessions[session])
                    context = get_context(center_idx,test_sessions[session])
                    word= word[np.newaxis,:,:,:]
                    context = context [np.newaxis, :, :, :]
                    feed_dict = {model.input_words: word,
                                 model.input_context: context}
                    actions_prob = sess.run(model.supervised_logits, feed_dict)
                    #print(np.argmax(actions_prob),event_goal)
                    cnf_matrix[event_goal,np.argmax(actions_prob)] +=1
                    #print(cnf_matrix)
                    #print(np.round(cnf_matrix/ (np.sum(cnf_matrix, axis=1)+ 1e-4),2))
        os_utils.touch_dir(save_result_dir)
        os_utils.pkl_write(save_result_dir + '/cnf_mat.pkl', cnf_matrix)
    else:
        cnf_matrix = os_utils.pkl_read(cnf_mat_path)



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

    os_utils.txt_write(save_result_dir +'/stats.txt',stats )




