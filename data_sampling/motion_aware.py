import sys
sys.path.append('../')
import imageio;
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyflow
import utils
import pickle
import os
import multiprocessing as mp
import traceback

def get_valid_frame_range(total_num_frames):
    invalid_trim_borders = True

    ## Make sure the center frame has roughly 1 min (30 frames) before and after it.
    ## For smaller videos, I have to decrease such limit
    num_frames = 3;
    step_btwn_frames = 12;

    while invalid_trim_borders:
        trim_borders = num_frames * step_btwn_frames + 1;
        left_extreme = trim_borders - num_frames * step_btwn_frames + 1
        right_extreme = trim_borders + num_frames * step_btwn_frames + 1
        if (left_extreme < 0 or right_extreme >= total_num_frames or (total_num_frames- 2 * trim_borders - 1) == 0):
            step_btwn_frames -=1
            invalid_trim_borders = True
        else:
            invalid_trim_borders = False

    return trim_borders,step_btwn_frames

def sample_high_motion_frames(vid,flow_mag,vdx_path=None,n=1):

    try:
        ## Default params
        total_num_frames = vid._meta['nframes']
        if (total_num_frames < 60):
            print('Video is too small',vdx_path)
            return None, None


        stack_diff_frame_idz = np.zeros(6, dtype=np.int32);
        trim_borders, step_btwn_frames = get_valid_frame_range(total_num_frames)
        center_frame_idx = int(np.random.choice(flow_mag.shape[0], n, p=flow_mag))
        center_frame_idx += trim_borders

        ideal_sampling_space = 15; ## Just an intuition from Shuffle & Learn

        left_sampling_space = min(ideal_sampling_space ,center_frame_idx //3) ## I need three frames from the left
        right_sampling_space = min(ideal_sampling_space, (total_num_frames-center_frame_idx)// 3)  ## I need three frames from the left

        current_sampling_space = min(left_sampling_space ,right_sampling_space );
        #print('Number of frames ', total_num_frames,current_sampling_space )
        left_ptr = center_frame_idx - current_sampling_space * 3 + current_sampling_space//2
        for i in range(6):
            stack_diff_frame_idz[i] = left_ptr  + i * current_sampling_space

            # stack_diff_frame_idz [1:3] = center_frame_idx - np.random.choice(np.arange(3 ,2 * step_btwn_frames),2,replace=False)#np.random.randint(2 ,2 * step,2,replace=False);
        # stack_diff_frame_idz[3:5] = center_frame_idx + np.random.choice(np.arange(3 ,2 * step_btwn_frames),2,replace=False) #np.random.randint(2 , 2 * step, 2);
        # stack_diff_frame_idz[0] = center_frame_idx - 2* step_btwn_frames - np.random.randint(3, step_btwn_frames);
        # stack_diff_frame_idz[5] = center_frame_idx + 2* step_btwn_frames + np.random.randint(3, step_btwn_frames);
        # stack_diff_frame_idz.sort()
        return center_frame_idx,stack_diff_frame_idz;
    except:
        traceback.print_exc()
        print('Something is wrong with sample_high_motion_frames',vdx_path)
        return None,None

def video2motion_level(vid,queue):
    fps = vid.get_meta_data()['fps']
    num_frames = vid._meta['nframes']

    trim_borders, step_btwn_frames = get_valid_frame_range(num_frames)
    flow_mag = np.zeros(num_frames- 2 * trim_borders - 1,dtype=np.float32);
    print(flow_mag.shape)
    #print(fps, num_frames)
    try:
        previous_frame = vid.get_data(trim_borders)

        # Flow Options:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
        colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        flow_mag_idx = 0
        for frame_idx in np.arange(trim_borders+1, num_frames-trim_borders):
            current_frame = vid.get_data(frame_idx)
            if np.array_equal(current_frame, previous_frame):
                ## If both frames are identical, optical flow is zero
                flow_mag[flow_mag_idx] = 0;
            else:

                im1 = previous_frame.astype(float) / 255.
                im2 = current_frame.astype(float) / 255.

                u, v, im2W = pyflow.coarse2fine_flow(
                    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                    nSORIterations, colType)
                flow = np.concatenate((u[..., None], v[..., None]), axis=2)

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                flow_mag[flow_mag_idx] = np.linalg.norm(mag);
            flow_mag_idx +=1
            previous_frame = current_frame;
    except:
        queue.put(None)
        return None
    ## Visualization code
    #plt.bar(np.arange(trim_borders+1, num_frames-trim_borders), flow_mag);
    #plt.savefig('freq.png')

    queue.put(flow_mag / np.sum(flow_mag))


def generate_flow_mag(path):
    dirs = utils.get_dirs(path);
    save_path = path + '_pkls'
    ## Fork 25 parallel threads to
    # process mutliple files at the same time
    queue_size = 25;
    queues = [None] * queue_size
    processes = [None] * queue_size
    filenames = [None] * queue_size
    idx = 0
    for d in dirs:
        current_dir = os.path.join(path,d);
        files = utils.get_files(current_dir,'.avi',append_base=True);
        for vdx_path in files:
            save_dir = os.path.join(save_path,d);
            if(not os.path.exists(save_dir)):
                os.makedirs(save_dir)

            name, ext = utils.get_file_name_ext(vdx_path);
            file_name = os.path.join(save_dir, name) + '.pkl';
            if(not os.path.exists(file_name)):
                print('processing ',vdx_path )
                vid = imageio.get_reader(vdx_path, 'ffmpeg')
                queues[idx] = mp.Queue()
                filenames[idx] = file_name ;
                processes[idx] = mp.Process(target=video2motion_level, args=(vid,queues[idx]))
                processes[idx].start()
                idx += 1
            else:
                flow_mag = utils.pkl_read(file_name)
                ## Make sure it contains a least one frame
                if(flow_mag.shape[0] == 0):
                    vid = imageio.get_reader(vdx_path, 'ffmpeg')
                    queues[idx] = mp.Queue()
                    filenames[idx] = file_name;
                    processes[idx] = mp.Process(target=video2motion_level, args=(vid, queues[idx]))
                    processes[idx].start()
                    idx += 1


            if(idx == queue_size):
                for i in range(queue_size):
                    processes[i].join()
                    flow_mag = queues[i].get()
                    if(flow_mag is not None):
                        utils.pkl_write(filenames[i],flow_mag)
                        print('Done ', filenames[i])
                    else:
                        print('Error in ', vdx_path)

                idx = 0;

    for i in range(idx):
        processes[i].join()
        flow_mag = queues[i].get()
        if(flow_mag is not None):
            utils.pkl_write(filenames[i],flow_mag)
        print('Done ', filenames[i])


if __name__ == '__main__':
    dataset_path = utils.get_dataset_path('UCF50')
    generate_flow_mag(dataset_path);

