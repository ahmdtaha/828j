import sys
sys.path.append('../')
import utils
import os
import imageio;
import numpy as np
if __name__ == '__main__':


    read_dir = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucf101/test_tuples_py36_r_chunks'
    activities = utils.get_dirs(read_dir);

    tuple_idx = 0;
    save_dir = '/vulcan/scratch/ahmdtaha/UCF101_tuples_class/test'
    utils.touch_dir(save_dir)
    lbls_file = os.path.join(save_dir, 'lbl.pkl')
    total_number_tuples = 0;

    with open(os.path.join('/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucf101', 'activities_list'), 'r') as f:
        activities_idxs = [x.strip() for x in f.readlines()]

    for activity in activities:
        current_dir = os.path.join(read_dir, activity)
        files = utils.get_files(current_dir, extension='pkl', append_base=False);
        for f in files:
            full_path = os.path.join(current_dir, f)
            tuples = utils.pkl_read(full_path)
            total_number_tuples += len(tuples[0])

    print('Excepted to has tuples ',total_number_tuples)

    lbls_ary = np.ones(total_number_tuples, dtype=np.int32) * -1;  ## Invalid Activity

    for activity in activities:
        current_dir = os.path.join(read_dir,activity)
        files = utils.get_files(current_dir,extension='pkl',append_base=False);
        for f in files:
            full_path = os.path.join(current_dir,f)
            tuples = utils.pkl_read(full_path)
            for i in range(len(tuples[0])):
                center_frame = tuples[0][i];
                stack_of_diff =tuples[1][i];
                imageio.imwrite(save_dir + '/frame' + "%07d" % (tuple_idx) + '.jpg', center_frame );
                utils.pkl_write(save_dir + '/frame' + "%07d" % (tuple_idx) + '.pkl', stack_of_diff)
                activity_id = activities_idxs.index(activity.split('_')[0])
                lbls_ary[tuple_idx] = activity_id;
                tuple_idx+=1;

    utils.pkl_write(lbls_file, lbls_ary);

