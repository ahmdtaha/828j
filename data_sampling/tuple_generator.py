import sys
sys.path.append('../../')
import utils
import data_sampling.motion_aware as motion_aware
import os
import cv2
import imageio
import constants as const
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_stack_diff(vid,stack_diff_frame_idz):

    unique_els = np.unique(stack_diff_frame_idz)
    if(unique_els.shape != stack_diff_frame_idz.shape):
        print("Something is wrong")

    num_output_frames= len(stack_diff_frame_idz)-1;
    stacked_diff = np.zeros((const.frame_height, const.frame_width,num_output_frames ), dtype=np.float32)

    for i in range(num_output_frames):
        current_frame = vid.get_data(stack_diff_frame_idz[i])
        current_frame = cv2.cvtColor(current_frame , cv2.COLOR_RGB2GRAY)
        current_frame = cv2.resize(current_frame ,(const.frame_height, const.frame_width))

        next_frame = vid.get_data(stack_diff_frame_idz[i+1])
        next_frame = cv2.cvtColor(next_frame , cv2.COLOR_RGB2GRAY)
        next_frame = cv2.resize(next_frame , (const.frame_height, const.frame_width))

        diff = next_frame.astype(np.float32) - current_frame.astype(np.float32)
        stacked_diff[:, :, i] = diff


    return stacked_diff

def sample_video_2(dataset_path,subset_files):
    sample = np.random.choice(subset_files);
    slash_index= sample.index('/');
    activity = sample[:slash_index]
    file_name = os.path.join(dataset_path,sample)
    #print(activity,file_name )
    return activity,file_name


def sample_video(dataset_path):
    dirs = utils.get_dirs(dataset_path);
    print(len(dirs))
    sample_activity = np.random.choice(len(dirs))
    d = dirs[sample_activity];
    current_dir = os.path.join(dataset_path, d);
    files = utils.get_files(current_dir, '.avi', append_base=True);
    sample_vdx = np.random.choice(len(files))
    print(len(files))
    current_vdx = files[sample_vdx]
    return d,current_vdx;
if __name__ == '__main__':
    dataset_path = utils.get_dataset_path('UCF50')
    pkls_path = dataset_path+'_pkls'
    save_dir = dataset_path+'_tuples_class'

    files_list = None

    ## Take one arg to indicate train or val. Train 0, Val 1
    if(len(sys.argv) == 2):
        current_subset = int(sys.argv[1])
    else:
        current_subset = 0

    if(current_subset == const.Subset.TRAIN.value):
        files_list = utils.txt_read(os.path.join(dataset_path+'_lists','trainlist.txt'))
        print('**** Train has ', len(files_list))
        save_dir = os.path.join(save_dir,'train')
        max_num_tuplex = 200000
    elif(current_subset == const.Subset.VAL.value):
        files_list = utils.txt_read(os.path.join(dataset_path + '_lists', 'vallist.txt'))
        print('**** Val has ', len(files_list))
        save_dir = os.path.join(save_dir, 'val')
        max_num_tuplex = 20000
    elif(current_subset == const.Subset.TEST.value):
        files_list = utils.txt_read(os.path.join(dataset_path + '_lists', 'testlist.txt'))
        print('*** Test has ', len(files_list))
        save_dir = os.path.join(save_dir, 'test')


    utils.touch_dir(save_dir)
    tuple_idx = 0
    tuple_idx = utils.last_tuple_idx(save_dir)
    #activity_list = ['Biking']
    activity_list = sorted(utils.get_dirs(dataset_path));
    print('activity_list ',len(activity_list),activity_list)
    #activity_list = ['Play_with_ball','Car','Drink','Feed','Look_at_Left','Look_at_Right','Pet','Shake','Sniff','Walk']
    lbls_file = os.path.join(save_dir,'lbl.pkl')

    if(tuple_idx == 0):
        lbls_ary = np.zeros(max_num_tuplex);
    else:
        lbls_ary = utils.pkl_read(lbls_file)
        print(lbls_ary)

    while tuple_idx < max_num_tuplex :
        #activity, current_vdx = sample_video(dataset_path)
        activity, current_vdx = sample_video_2(dataset_path,files_list)
        activity_lbl = activity_list.index(activity)
        lbls_ary[tuple_idx] = activity_lbl;
        #print(activity_lbl )
        vdx_path = os.path.join(current_vdx)
        name, ext = utils.get_file_name_ext(vdx_path);
        vid = imageio.get_reader(vdx_path, 'ffmpeg')

        pkl_file = os.path.join(pkls_path,activity,name)+'.pkl'
        flow_mag = utils.pkl_read(pkl_file)

        center_frame_idx, stack_diff_frame_idz = motion_aware.sample_high_motion_frames(vid, flow_mag,vdx_path)

        if(center_frame_idx is not None):
            center_img = vid.get_data(center_frame_idx)
            center_img = cv2.resize(center_img , (const.frame_height, const.frame_width))

            imageio.imwrite(save_dir+ '/frame' + "%07d" % (tuple_idx) + '.jpg',center_img );
            stack_diff = create_stack_diff(vid,stack_diff_frame_idz)
            utils.pkl_write(save_dir+ '/frame' + "%07d" % (tuple_idx) + '.pkl',stack_diff)

            utils.pkl_write(lbls_file,lbls_ary);
            visualize = False
            if(visualize):
                for i in range(stack_diff.shape[2]):
                    plt.imshow(stack_diff[:, :, i]);
                    plt.savefig(save_dir + '/frame_' + '%07d' % (tuple_idx) +'_'+ str(i)+'_pkl.jpg')

            tuple_idx +=1
            if(tuple_idx % 100 == 0):
                print('Number of tuples ', tuple_idx, '/ ', max_num_tuplex)
                print(activity, current_vdx )

