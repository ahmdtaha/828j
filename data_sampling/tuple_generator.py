import sys
sys.path.append('../')
import data_sampling.motion_aware as motion_aware
import utils
import os
import cv2
import constants as const
import configuration as file_const
import numpy as np
import imageio
import multiprocessing as mp
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_stack_diff(vid,stack_diff_frame_idz):

    unique_els = np.unique(stack_diff_frame_idz)
    if(unique_els.shape != stack_diff_frame_idz.shape):
        print("Something is wrong")

    num_output_frames= len(stack_diff_frame_idz);
    stacked_diff = np.zeros((const.nominal_height,const.nominal_width,const.frame_channels,num_output_frames), dtype=np.uint8)

    for i in range(num_output_frames):
        current_frame = vid.get_data(stack_diff_frame_idz[i])
        if(current_frame.shape[0] != const.nominal_height or current_frame.shape[1] != const.nominal_width):
            current_frame = cv2.resize(current_frame, (const.nominal_width,const.nominal_height))

        stacked_diff[:, :, :,i] = current_frame


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

def save_tuple(vid,center_frame_idx,stack_diff_frame_idz,save_dir,tuple_idx,lbls_ary,activity_lbl,queue=None):
    try:
        center_img = vid.get_data(center_frame_idx)
        if (center_img.shape[0] != const.nominal_height or center_img.shape[1] != const.nominal_width):
            center_img = cv2.resize(center_img, (const.nominal_height, const.nominal_width))

        imageio.imwrite(save_dir + '/frame' + "%07d" % (tuple_idx) + '.jpg', center_img);
        stack_diff = create_stack_diff(vid, stack_diff_frame_idz)
        utils.pkl_write(save_dir + '/frame' + "%07d" % (tuple_idx) + '.pkl', stack_diff)



        visualize = False
        if (visualize):
            for i in range(stack_diff.shape[3]):
                cv2.imwrite(save_dir + '/frame_' + '%07d' % (tuple_idx) + '_' + str(i) + '_pkl.jpg',
                            cv2.cvtColor(stack_diff[:, :, :, i], cv2.COLOR_RGB2BGR))
                # plt.imshow(stack_diff[:, :, :,i]);
                # plt.axis('off')
                # plt.savefig(save_dir + '/frame_' + '%07d' % (tuple_idx) +'_'+ str(i)+'_pkl.jpg')
        queue.put(True)
    except:
        pass
        queue.put(False)
        queue.put(tuple_idx)


if __name__ == '__main__':
    dataset_path = utils.get_dataset_path(file_const.dataset_name)
    pkls_path = dataset_path+'_pkls'
    save_dir = dataset_path+'_tuples_class'

    files_list = None

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--is_training', action='store_true')

    args = parser.parse_args()

    ## Take one arg to indicate train or val. Train 0, Val 1
    if(args.is_training):
        current_subset = 0 ## Train
    else:
        current_subset = 1 ## Val

    if(current_subset == const.Subset.TRAIN.value):
        files_list = utils.txt_read(os.path.join(dataset_path+'_lists','trainlist.txt'))
        print('**** Train has ', len(files_list))
        save_dir = os.path.join(save_dir,'train')
        max_num_tuplex = 2000 ##200000
    elif(current_subset == const.Subset.VAL.value):
        files_list = utils.txt_read(os.path.join(dataset_path + '_lists', 'vallist.txt'))
        print('**** Val has ', len(files_list))
        save_dir = os.path.join(save_dir, 'val')
        max_num_tuplex =  100 ## 20000
    elif(current_subset == const.Subset.TEST.value):
        files_list = utils.txt_read(os.path.join(dataset_path + '_lists', 'testlist.txt'))
        print('*** Test has ', len(files_list))
        save_dir = os.path.join(save_dir, 'test')


    utils.touch_dir(save_dir)
    tuple_idx = 0
    tuple_idx = max(0,utils.last_tuple_idx(save_dir)  - 5) ## As caution, regenerate last 5 tuples
    activity_list = sorted(utils.get_dirs(dataset_path));
    print('activity_list ',len(activity_list),activity_list)

    lbls_file = os.path.join(save_dir,'lbl.pkl')

    if(tuple_idx == 0):
        lbls_ary = np.ones(max_num_tuplex,dtype=np.int32)* -1; ## Invalid Activity
    else:
        lbls_ary = utils.pkl_read(lbls_file)
        tuple_idx = 0;

    import time
    start_time = time.time()

    queue_size = 25;
    queues = [None] * queue_size
    processes = [None] * queue_size
    idx = 0
    while tuple_idx< max_num_tuplex:
        if ( lbls_ary[tuple_idx] != -1):
            tuple_idx +=1
            continue;
        #activity, current_vdx = sample_video(dataset_path)
        activity, current_vdx = sample_video_2(dataset_path,files_list)
        activity_lbl = activity_list.index(activity)

        #print(activity_lbl )
        vdx_path = os.path.join(current_vdx)
        name, ext = utils.get_file_name_ext(vdx_path);
        try:
            vid = imageio.get_reader(vdx_path, 'ffmpeg')
        except:
            traceback.print_exc()
            continue

        pkl_file = os.path.join(pkls_path,activity,name)+'.pkl'
        flow_mag = utils.pkl_read(pkl_file)

        center_frame_idx, stack_diff_frame_idz = motion_aware.sample_high_motion_frames(vid, flow_mag,vdx_path)
        if(center_frame_idx is not None):

            # save_tuple(vid, center_frame_idx, stack_diff_frame_idz, save_dir, tuple_idx, lbls_ary, activity_lbl)
            # tuple_idx += 1
            queues[idx] = mp.Queue()
            processes[idx] = mp.Process(target=save_tuple, args=(vid, center_frame_idx, stack_diff_frame_idz, save_dir,tuple_idx,lbls_ary,activity_lbl,queues[idx]))
            processes[idx].start()
            lbls_ary[tuple_idx] = activity_lbl;
            idx +=1;
            tuple_idx +=1

            if (idx == queue_size):
                for i in range(queue_size):
                    processes[i].join()
                    success = queues[i].get()
                    if (not success):
                        failure_tuple_idx = queues[i].get()
                        lbls_ary[failure_tuple_idx] = -1
                        print('Something went wrong')
                idx = 0;
                utils.pkl_write(lbls_file, lbls_ary);



            if(tuple_idx % 100 == 0):
                print('Number of tuples ', tuple_idx, '/ ', max_num_tuplex)
                print(activity, current_vdx )

    for i in range(idx):
        processes[i].join()
        success = queues[i].get()
        if (not success):
            print('Something went wrong')
            failure_tuple_idx = queues[i].get()
            lbls_ary[failure_tuple_idx] = -1

    utils.pkl_write(lbls_file, lbls_ary);
    for i in range(10):
        print(activity_list[lbls_ary[i]],end="\t")



    elapsed_time = time.time() - start_time
    print('elapsed_time :', elapsed_time)