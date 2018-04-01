import sys
sys.path.append('../')
import os
import configuration as config
import utils.os_utils as os_utils
import imageio
import traceback
import pandas as pd

def extract_frames():
    original_db = '/Users/ahmedtaha/Documents/dataset/hmdb51_org'
    sampled_db = '/Users/ahmedtaha/Documents/dataset/hmdb51_sampled'

    os_utils.touch_dir(sampled_db);

    activities = os_utils.get_dirs(original_db)

    video_names, video_lbl, video_len = [], [], []
    for activity in activities:
        activity_path = os.path.join(original_db, activity, activity);
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

            video_names.append(video_name)
            video_lbl.append(acitiviy_lbl)
            video_len.append(num_frames_saved)
            print(video_name, ' ', acitiviy_lbl, ' ', num_frames_saved)

    dataset_summary = pd.DataFrame({'video-name': video_names,
                                    'video-lbl': video_lbl,
                                    'video-len': video_len})
    dataset_summary.to_pickle(sampled_db + '/db_summary.pkl')
    dataset_summary.to_csv(sampled_db + '/db_summary.csv')
if __name__ == '__main__':
    sampled_db = '/Users/ahmedtaha/Documents/dataset/hmdb51_sampled'
    dataset_summary = pd.read_pickle(sampled_db  + '/db_summary.pkl')
    print(dataset_summary.shape)

    valid_dataset_summary = dataset_summary[dataset_summary ['video-len'] >= 7]
    print(valid_dataset_summary.shape)
    valid_dataset_summary = valid_dataset_summary.sample(frac=1)
    print(valid_dataset_summary.count)
    train = int(0.6 * valid_dataset_summary.shape[0])
    val = int(0.1 * valid_dataset_summary.shape[0])

    train_set = valid_dataset_summary.iloc[:train]
    val_set = valid_dataset_summary.iloc[train:train+val]
    test_set = valid_dataset_summary.iloc[train + val:]

    train_set.to_pickle(sampled_db + '/train_db.pkl')
    train_set.to_csv(sampled_db + '/train_db.csv')

    val_set.to_pickle(sampled_db + '/val_db.pkl')
    val_set.to_csv(sampled_db + '/val_db.csv')

    test_set.to_pickle(sampled_db + '/test_db.pkl')
    test_set.to_csv(sampled_db + '/test_db.csv')










