import sys
sys.path.append('../')
import os
import configuration as config
import utils.os_utils as os_utils
import imageio
import traceback
import pandas as pd
import numpy as np
import logging
logging.basicConfig(filename='../dump/ucf_prepare.log',level=logging.DEBUG)

def extract_frames():
    original_db = '/Users/ahmedtaha/Documents/dataset/UCF101'
    sampled_db = '/Users/ahmedtaha/Documents/dataset/UCF101_sampled'

    os_utils.touch_dir(sampled_db);

    activities = os_utils.get_dirs(original_db)

    video_names, video_lbl, video_len = [], [], []
    for activity in activities:
        activity_path = os.path.join(original_db, activity);
        acitiviy_lbl = activities.index(activity)
        videos_paths = os_utils.get_files(activity_path, '.avi', append_base=True);

        for video_path in videos_paths:
            video_name, _ = os_utils.get_file_name_ext(video_path);
            video_save_path = os.path.join(sampled_db, video_name)
            os_utils.touch_dir(video_save_path)
            num_frames_saved = 0
            try:
                vid = imageio.get_reader(video_path, 'ffmpeg')
                total_num_frames = vid._meta['nframes']
                step = 10;  ## 3 frames per second
                for frame_idx in range(0, total_num_frames, step):
                    current_frame = vid.get_data(frame_idx)
                    frame_save_path = os.path.join(video_save_path, 'frame_%04d' % (num_frames_saved) + '.jpg');
                    print(frame_save_path)
                    imageio.imwrite(frame_save_path, current_frame)
                    num_frames_saved += 1


            except:
                traceback.print_exc()
                logging.info('Review File ',video_save_path)

            video_names.append(video_name)
            video_lbl.append(acitiviy_lbl)
            video_len.append(num_frames_saved)
            print(video_name, ' ', acitiviy_lbl, ' ', num_frames_saved)

    dataset_summary = pd.DataFrame({'video-name': video_names,
                                    'video-lbl': video_lbl,
                                    'video-len': video_len})
    dataset_summary.to_pickle(sampled_db + '/db_summary.pkl')
    dataset_summary.to_csv(sampled_db + '/db_summary.csv')

def divide():
    sampled_db = '/Users/ahmedtaha/Documents/dataset/hmdb51_sampled'
    dataset_summary = pd.read_pickle(sampled_db + '/db_summary.pkl')
    print(dataset_summary.shape)

    valid_dataset_summary = dataset_summary[dataset_summary['video-len'] >= 7]
    print(valid_dataset_summary.shape)
    valid_dataset_summary = valid_dataset_summary.sample(frac=1)
    print(valid_dataset_summary.count)
    train = int(0.6 * valid_dataset_summary.shape[0])
    val = int(0.1 * valid_dataset_summary.shape[0])

    train_set = valid_dataset_summary.iloc[:train]
    val_set = valid_dataset_summary.iloc[train:train + val]
    test_set = valid_dataset_summary.iloc[train + val:]

    train_set.to_pickle(sampled_db + '/train_db.pkl')
    train_set.to_csv(sampled_db + '/train_db.csv')

    val_set.to_pickle(sampled_db + '/val_db.pkl')
    val_set.to_csv(sampled_db + '/val_db.csv')

    test_set.to_pickle(sampled_db + '/test_db.pkl')
    test_set.to_csv(sampled_db + '/test_db.csv')

def define_splits():
    splits_path = '/Users/ahmedtaha/Documents/dataset/ucfTrainTestlist'

    sampled_db = '/Users/ahmedtaha/Documents/dataset/UCF101_sampled'
    dataset_summary = pd.read_pickle(sampled_db + '/db_summary.pkl')
    num_files = dataset_summary.shape[0];
    splits = np.zeros((num_files, 3))

    dataset_summary.to_pickle(sampled_db + '/db_summary_splits.pkl')
    dataset_summary.to_csv(sampled_db + '/db_summary_splits.csv')
    dataset_summary['split1'] = splits[:, 0];
    dataset_summary['split2'] = splits[:, 1];
    dataset_summary['split3'] = splits[:, 2];

    # dataset_summary.to_pickle(sampled_db + '/db_summary_splits.pkl')
    # dataset_summary.to_csv(sampled_db + '/db_summary_splits.csv')

    for i in range(1, 4):
        current_split = 'list0%d' % i + '.txt'

        files = os_utils.get_files(splits_path, append_base=True, extension=current_split)
        train_set =1;
        test_set = 2;
        print(files)
        test_file = files[0]
        train_file = files[1]
        test_lines = os_utils.txt_read(test_file)
        train_lines = os_utils.txt_read(train_file)
        for line in test_lines:
            name = os_utils.get_last_part(line)
            name = os_utils.get_file_name_ext(name)[0]
            dataset_summary.loc[dataset_summary['video-name'] == name, 'split' + str(i)] = test_set

        for line in train_lines:
            name, label = line.split()
            name = os_utils.get_last_part(name)
            name = os_utils.get_file_name_ext(name)[0]
            dataset_summary.loc[dataset_summary['video-name'] == name, 'split' + str(i)] = train_set
            #print(dataset_summary.loc[dataset_summary['video-name'] == name])
            #break;
        #break;

        # for file in files:
        #     act_vdz = os_utils.txt_read(file)
        #     for act_vdx in act_vdz:
        #         name, id = act_vdx.split()
        #         name = os_utils.get_file_name_ext(name)[0]
        #         print('Name ', name)
        #         print('ID ', id)
        #         # row = dataset_summary.loc[dataset_summary['video-name'] == name]
        #         # row['split'+str(i)] = id
        #         dataset_summary.loc[dataset_summary['video-name'] == name, 'split' + str(i)] = id
        #         # print(dataset_summary.loc[dataset_summary['video-name'] == name])

    dataset_summary.to_pickle(sampled_db + '/db_summary_splits.pkl')
    dataset_summary.to_csv(sampled_db + '/db_summary_splits.csv')
if __name__ == '__main__':
    define_splits();














