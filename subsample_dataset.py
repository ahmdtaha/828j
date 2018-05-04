import cv2
import glob
import imageio
import os
import os.path as osp
import pickle
import traceback


def downsample_video(video_path, outpath, sampling_step=5):
    try:
        video = imageio.get_reader(video_path, 'ffmpeg')

        total_num_frames = video._meta['nframes']
        num_sampled_frames = (total_num_frames // sampling_step) + (
            1 if total_num_frames % sampling_step > 0 else 0)
        frames = [None] * num_sampled_frames

        # with open(sampled_video_path, 'rb') as f:
        #     fs = pickle.load(f)
        # for i in range(len(fs)):
        #     save_path = k_output_dump_path + '/%d.png' % i
        #     frame = fs[i]
        #     img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(save_path, img)
        #     print('saved frame %d to %s' % (frame_index, save_path))

        # for frame_index in range(2, total_num_frames, sampling_step):
        # for frame_index in range(0, total_num_frames - 2 * sampling_step, sampling_step):
        # for frame_index in range(0, total_num_frames - 3 * sampling_step, sampling_step):
        for frame_index in range(0, total_num_frames, sampling_step):
            # print('%d / %d' % (frame_index, total_num_frames))
            frame = video.get_data(frame_index)
            frames[frame_index // sampling_step] = frame

        with open(outpath, 'wb') as f:
            pickle.dump(frames, f)
    except:
        # traceback.print_exc()
        print('Error: failed to read video %s\n' % video_path)
        return


if __name__ == '__main__':
    k_base_dir = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning'
    dataset_path = osp.join(k_base_dir, 'datasets', 'ucf101')
    activities_filepath = osp.join(dataset_path, 'activities_list')
    dataset_out_path = osp.join(k_base_dir, 'datasets', 'ucf101_downsampled')

    with open(activities_filepath, 'r') as f:
        activities = [x.strip() for x in f.readlines()]

    assert len(activities) == 101, 'Error parsing the UCF101 dataset'

    for idx, activity in enumerate(activities):
        print('Processing activity #%d: %s' % (idx + 1, activity))
        video_paths = glob.glob(osp.join(dataset_path, activity, '*.avi'))
        outdir = osp.join(dataset_out_path, activity)
        os.makedirs(outdir, exist_ok=True)

        for video_path in video_paths:
            video_name = osp.basename(video_path)
            video_out_path = osp.join(outdir, video_name[:-4] + '.pkl')
            if not osp.isfile(video_out_path):
                print('Processing %s' % video_path)
                downsample_video(video_path, video_out_path, sampling_step=5)
