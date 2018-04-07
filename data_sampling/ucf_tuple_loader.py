import sys
sys.path.append('../')
import os
import configuration as config
import data_sampling.data_args as data_args
import constants as const
import numpy as np
import cv2
from utils import os_utils
import traceback
import imageio
import data_sampling.honda_labels as honda_lbls
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import multiprocessing as mp
from multiprocessing import Pool,TimeoutError
import pandas as pd
from datetime import datetime

val_dir = 'val'


class UCFTupleLoader:

    def load_subset(self,subset):
        df = pd.read_pickle(config.db_path + '/db_summary_splits.pkl')
        #print(df.shape)
        split = config.db_split

        # The video is included in the training set if id is 1
        # The video is included in the testing set if id is 2
        if(subset == 'train'):
            sub_df = df.loc[ (df['split'+str(split)] == 1) & (df['video-len'] >= 7)] ##
        elif (subset == 'test'):
            sub_df = df.loc[(df['split' + str(split)] == 2) & (df['video-len'] >= 7)]

        #print(sub_df.shape)

        videos_names = sub_df['video-name'].tolist()
        videos_lbls = sub_df['video-lbl'].tolist()
        videos_len = sub_df['video-len'].tolist()

        return videos_names , videos_lbls,videos_len



    def __init__(self, args):
        print(self.__class__.__name__)
        # There UCF dataset doesn't have clear validation set
        if (config.dataset_name != 'UCF101'):
            print('Something is wrong with dataset, double check the config')
            sys.exit(1)

        total_train_videos, total_train_videos_lbls, total_train_videos_len = self.load_subset('train');

        ## TODO: bad quick workaround to fix train and val permutation
        np.random.seed(828);

        num_files = len(total_train_videos)
        train_set_size = int(num_files  * 0.8)
        samples_permutated = np.random.permutation(num_files);
        train_idx = samples_permutated[0:train_set_size]
        val_idx = samples_permutated[train_set_size:]
        # [1466 2151  270 ...,  361 2272 2411]
        print(train_idx )
        self.train_videos = [total_train_videos[i] for i in train_idx]
        self.train_videos_lbls = [total_train_videos_lbls[i] for i in train_idx]
        self.train_videos_len = [total_train_videos_len[i] for i in train_idx]
        self.val_videos = [total_train_videos[i] for i in val_idx]
        self.val_videos_lbls = [total_train_videos_lbls[i] for i in val_idx]
        self.val_videos_len = [total_train_videos_len[i] for i in val_idx]



        print(datetime.now())
        import time

        np.random.seed(int(time.time()))
        print('Train ',len(self.train_videos_lbls), ' Val ',len(self.val_videos_lbls))


    def imgs2sod(self,imgs,ordered=True):

        rand_crop = np.random.rand();
        y = int(rand_crop * (imgs[0].shape[0] - imgs[0].shape[0] * 0.8))
        x = int(rand_crop * (imgs[0].shape[1] - imgs[0].shape[1] * 0.8))

        rand_rgb_channel = np.random.choice(3);
        stack_diff = np.zeros((const.frame_height, const.frame_width, const.context_channels))

        if (ordered):
            frames_order = np.arange(const.context_channels + 1)
        else:
            frames_order = np.random.permutation(const.context_channels + 1)

        for i in range(const.context_channels):
            current_frame = imgs[frames_order[i]][y:y+int(imgs[0].shape[0] * 0.8), x:x+int(imgs[0].shape[1] * 0.8),rand_rgb_channel];
            next_frame = imgs[frames_order[i+1]][y:y+int(imgs[0].shape[0] * 0.8), x:x+int(imgs[0].shape[1] * 0.8), rand_rgb_channel];

            current_frame = cv2.resize(current_frame, (const.frame_height, const.frame_width))
            next_frame = cv2.resize(next_frame, (const.frame_height, const.frame_width))

            stack_diff[:,:,i] = current_frame.astype(np.int32) - next_frame.astype(np.int32);

        return stack_diff;

    def get_context(self,center_idx,session,ordered=True):
        imgs = []
        for idx in [-3,-2,-1,1,2,3]:
            img_path = os.path.join(config.db_path, session, 'frame_%04d.jpg' % (center_idx+idx));
            img = imageio.imread(img_path);
            imgs.append(img)
        return self.imgs2sod(imgs,ordered)


    def augment_img(self,im):
        rand_crop = np.random.rand();
        rand_rgb_channel = np.random.choice(3);
        # Randomly sample 80% of the image
        y = int(rand_crop * (im.shape[0] - im.shape[0] * 0.8))
        x = int(rand_crop * (im.shape[1] - im.shape[1] * 0.8))
        im = im[y:y+int(im.shape[0] * 0.8), x:x+int(im.shape[1] * 0.8), rand_rgb_channel];
        im = cv2.resize(im, (const.frame_height, const.frame_width))
        im = im[:, :, np.newaxis]
        im = np.repeat(im, 3, axis=2);
        return im

    def get_img(self,frame_idx,session):

        img_path = os.path.join(config.db_path,session,'frame_%04d.jpg' % frame_idx);
        img = imageio.imread(img_path);
        return self.augment_img(img)

    def load_pos_neg_tuple(self, pos_tuple,neg_tuple, frame_sampling_idx, batch_idx, current_sessions,current_sessions_len,
                       ordered=True):
        pos_session = current_sessions[pos_tuple]
        pos_video_len = current_sessions_len[pos_tuple]
        pos_center_idx = int(3 + frame_sampling_idx * (pos_video_len - 7));

        neg_session = current_sessions[neg_tuple]
        neg_video_len = current_sessions_len[neg_tuple]
        neg_center_idx = int(3 + frame_sampling_idx * (neg_video_len - 7));


        word = self.get_img(pos_center_idx, pos_session)
        context = self.get_context(neg_center_idx, neg_session, ordered)

        visualize = False
        if(visualize):
            if(ordered):
                label = 2;
                vis_sod(context, label, 'p_' + str(batch_idx), '_sod')
            else:
                label = 3;

                vis_sod(context, label, 'p_' + str(batch_idx), '_sod_u')
                tmp = self.get_context(neg_center_idx, neg_session, ordered=True) # Just to compare the ordered with unordered
                vis_sod(tmp, label, 'p_' + str(batch_idx), '_sod_o')

            vis_img(word, label, 'p_' + str(batch_idx), '_img')
            imgs = []
            for i in [-5,-3,-1,1,3,5]:
                img_path = os.path.join(config.honda_session_path, 'frames', neg_session, 'frame_%04d.jpg' % (neg_center_idx+i));
                img = imageio.imread(img_path);
                imgs.append(img)

            vis_frames(imgs,label, 'p_' + str(batch_idx), '_imgs')
        return word,context

    def load_temporal_tuple(self, pos_tuple, frame_sampling_idx, batch_idx, current_sessions, current_sessions_lbl,current_sessions_len,ordered):
        video = current_sessions[pos_tuple]
        video_len = current_sessions_len[pos_tuple]
        video_lbl = current_sessions_lbl[pos_tuple]
        center_idx = int(3 + frame_sampling_idx * (video_len - 7));
        imgs_list = [-3, -2, -1, 1, 2, 3];

        if(ordered != -1):
            imgs_list[ordered], imgs_list[ordered+1] = imgs_list[ordered+1],imgs_list[ordered];

        imgs = []
        for idx in imgs_list:
            img_path = os.path.join(config.hmdb_session_path, video, 'frame_%04d.jpg' % (center_idx + idx));

            img = imageio.imread(img_path);
            imgs.append(img)
        return self.imgs2sod(imgs, ordered)



    def load_pos_tuple(self, pos_tuple, frame_sampling_idx, batch_idx, current_sessions, current_sessions_lbl,current_sessions_len,ordered=True):
        video = current_sessions[pos_tuple]

        video_len = current_sessions_len[pos_tuple]
        video_lbl = current_sessions_lbl[pos_tuple]
        #print(video,video_lbl)
        center_idx = int(3 + frame_sampling_idx * (video_len -7));

        word = self.get_img(center_idx, video)
        context = self.get_context(center_idx, video , ordered)
        return word,context,video_lbl;

    def unsupervised_next_temporal(self,subset):


        if (subset == const.Subset.TRAIN):
            subset_size = len(self.train_videos)
            current_sessions = self.train_videos
            current_sessions_lbl = self.train_videos_lbls
            current_sessions_len = self.train_videos_len

        elif (subset == const.Subset.VAL):
            subset_size = len(self.val_videos)
            current_sessions = self.val_videos
            current_sessions_lbl = self.val_videos_lbls
            current_sessions_len = self.val_videos_len

        else:
            print('Only Train and val subset supported')
            sys.exit(1)

        pos_tuple = np.random.randint(low=0, high=subset_size, size=(const.batch_size));
        frame_sampling_idx= np.random.rand(const.batch_size);
        contexts = np.zeros((const.batch_size, const.frame_height, const.frame_width, const.context_channels))

        labels = np.zeros((const.batch_size), dtype=np.int32)

        sampling_ratio = np.random.randint(low=-1, high=const.context_channels, size=(const.batch_size));

        sample_count = np.zeros(4);
        for batch_idx in np.arange(0, const.batch_size):

            labels[batch_idx] = sampling_ratio[batch_idx]+1;
            context= self.load_temporal_tuple(pos_tuple[batch_idx], frame_sampling_idx[batch_idx],batch_idx ,
                                                     current_sessions,current_sessions_lbl,current_sessions_len,ordered=sampling_ratio[batch_idx]);

            contexts[batch_idx, :, :] = context

        #print(sample_count)
        labels_hot_vector = os_utils.hot_one_vector(labels,const.context_channels+1);

        return None, contexts, labels_hot_vector

    def unsupervised_next(self,subset):


        if (subset == const.Subset.TRAIN):
            subset_size = len(self.train_videos)
            current_sessions = self.train_videos
            current_sessions_lbl = self.train_videos_lbls
            current_sessions_len = self.train_videos_len

        elif (subset == const.Subset.VAL):
            subset_size = len(self.val_videos)
            current_sessions = self.val_videos
            current_sessions_lbl = self.val_videos_lbls
            current_sessions_len = self.val_videos_len
        else:
            print('Only Train and val subset supported')
            sys.exit(1)

        pos_tuple = np.random.randint(low=0, high=subset_size, size=(const.batch_size));
        frame_sampling_idx= np.random.rand(const.batch_size);
        neg_tuple = np.random.randint(low=0, high=subset_size, size=(const.batch_size));

        words = np.zeros((const.batch_size, const.frame_height, const.frame_width, const.frame_channels))
        contexts = np.zeros((const.batch_size, const.frame_height, const.frame_width, const.context_channels))

        labels = np.zeros((const.batch_size), dtype=np.int32)

        sampling_ratio = np.random.rand(const.batch_size);

        sample_count = np.zeros(4);
        for batch_idx in np.arange(0, const.batch_size):
            if (sampling_ratio[batch_idx] < 0.25):
                ## Same Image , Ordered
                labels[batch_idx] = 0;
                word,context,_ = self.load_pos_tuple(pos_tuple[batch_idx], frame_sampling_idx[batch_idx],batch_idx ,
                                                     current_sessions,current_sessions_lbl,current_sessions_len,ordered=True);
            elif (sampling_ratio[batch_idx] < 0.5):
                ## Same Image , UnOrdered
                labels[batch_idx] = 1;
                word, context,_ = self.load_pos_tuple(pos_tuple[batch_idx], frame_sampling_idx[batch_idx],batch_idx ,
                                                      current_sessions,current_sessions_lbl,current_sessions_len,ordered=False);
            elif (sampling_ratio[batch_idx] < 0.75):
                ## Different Image , Ordered
                labels[batch_idx] = 2;
                word, context = self.load_pos_neg_tuple(pos_tuple[batch_idx],neg_tuple[batch_idx],
                                                        frame_sampling_idx[batch_idx], batch_idx, current_sessions,
                                                        current_sessions_len, ordered=True);
            else:
                ## Different Image , UnOrdered
                labels[batch_idx] = 3;
                word, context = self.load_pos_neg_tuple(pos_tuple[batch_idx], neg_tuple[batch_idx],
                                                        frame_sampling_idx[batch_idx], batch_idx, current_sessions,
                                                        current_sessions_len, ordered=False);

            sample_count[labels[batch_idx]] +=1
            words[batch_idx, :, :] = word
            contexts[batch_idx, :, :] = context

        #print(sample_count)
        labels_hot_vector = os_utils.hot_one_vector(labels,4);
        return words, contexts, labels_hot_vector


    def mp_load_pos_tuple(self, pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations,
                          ordered, labels,words,contexts,queue):
        try:
            word, context, goal = self.load_pos_tuple(pos_tuple, annotation_idx, batch_idx, current_sessions,
                                                      current_sessions_annotations, ordered)

            print('word mean-1',np.mean(word))
            labels[batch_idx] = goal;
            words[batch_idx, :, :] = word
            print('word mean-2', np.mean(words))
            contexts[batch_idx, :, :] = context
            queue.put(True)  ## Success Flag
            # queue.put(word)
            # queue.put(context)
            # queue.put(goal)
        except:
            traceback.print_exc()
            queue.put(False)  ## Fail Flag

    def pool_load_pos_tuple(self, pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations,
                          ordered):
        try:
            word, context, goal = self.load_pos_tuple(pos_tuple, annotation_idx, batch_idx, current_sessions,
                                                      current_sessions_annotations, ordered)

            return (word, context, goal)

        except:
            traceback.print_exc()
            return (None,None,None)




    def supervised_next(self,subset):

        if (subset == const.Subset.TRAIN):
            subset_size = len(self.train_videos)
            current_sessions = self.train_videos
            current_sessions_lbl = self.train_videos_lbls
            current_sessions_len = self.train_videos_len

        elif (subset == const.Subset.VAL):
            subset_size = len(self.val_videos)
            current_sessions = self.val_videos
            current_sessions_lbl = self.val_videos_lbls
            current_sessions_len = self.val_videos_len
        else:
            print('Only Train and val subset supported')
            sys.exit(1)

        pos_tuple = np.random.randint(low=0, high=subset_size, size=(const.batch_size));
        frame_sampling_idx = np.random.rand(const.batch_size);

        words = np.zeros((const.batch_size, const.frame_height, const.frame_width, const.frame_channels))
        contexts = np.zeros((const.batch_size, const.frame_height, const.frame_width, const.context_channels))

        labels = np.zeros((const.batch_size), dtype=np.int32)

        sample_count = np.zeros(config.num_classes);
        #queue_size = const.batch_size;
        # queues = [None] * queue_size
        # processes = [None] * queue_size
        # print('queue_size ',queue_size)

        # queue_size = 5
        # pool = Pool(processes=queue_size )
        # res = [None] * queue_size
        for batch_idx in np.arange(0,const.batch_size):

            word, context, goal = self.load_pos_tuple(pos_tuple[batch_idx], frame_sampling_idx[batch_idx], batch_idx,
                                                   current_sessions, current_sessions_lbl, current_sessions_len,
                                                   ordered=True);
            labels[batch_idx] = goal;
            words[batch_idx, :, :] = word
            contexts[batch_idx, :, :] = context

            # sample_count[labels[batch_idx]] += 1
            # print(batch_idx , 'Goal ',goal,honda_lbls.honda_num2labels[goal])
            # for j in range(queue_size):
            #     res[j] = pool.apply_async(self.pool_load_pos_tuple,args=(pos_tuple, annotation_idx, batch_idx+j,
            #                                                                     current_sessions,current_sessions_annotations, True))
            #
            # for j in range(queue_size):
            #     try:
            #         result = res[j].get(timeout=10)
            #         word, context, goal = result;
            #         #print(batch_idx+j, 'Goal ', goal, honda_lbls.honda_num2labels[goal])
            #         #print('Worked for batch_idx ',batch_idx )
            #     except TimeoutError:
            #         print('TimeoutError')
            #         word, context, goal = self.load_pos_tuple(pos_tuple, annotation_idx, batch_idx, current_sessions,
            #                                                   current_sessions_annotations, ordered=True);
            #     labels[batch_idx+j] = goal;
            #     words[batch_idx+j, :, :] = word
            #     contexts[batch_idx+j, :, :] = context

            # queues[batch_idx] = mp.Queue()
        #     processes[batch_idx] = mp.Process(target=self.mp_load_pos_tuple,args=(pos_tuple, annotation_idx, batch_idx,
        #                                                                        current_sessions,current_sessions_annotations, True,labels,words,contexts,queues[batch_idx]))
        #     processes[batch_idx].start()
        #
        #
        #
        #
        # for batch_idx in range(const.batch_size):
        #     processes[batch_idx].join()
        #     success = queues[batch_idx].get()
        #     if (not success):
        #         word, context, goal = self.load_pos_tuple(pos_tuple, annotation_idx, batch_idx, current_sessions,current_sessions_annotations, ordered=True);
        #
        #
        #         labels[batch_idx] = goal;
        #         words[batch_idx, :, :] = word
        #         contexts[batch_idx, :, :] = context

        #print(sample_count)
        labels_hot_vector = os_utils.hot_one_vector(labels, config.num_classes);
        return words, contexts, labels_hot_vector

    def next(self, subset,supervised = False):
        if supervised :
            return self.supervised_next(subset)
        else:
            return self.unsupervised_next(subset)

def vis_img(img,label,prefix,suffix):
    cv2.imwrite(config.dump_path + prefix + '_' + str(label) + suffix + '.png',img)

def vis_sod(stack_diff,label,prefix,suffix):
    images = []
    for j in range(5):
        im = stack_diff[:, :, j];
        im = ((im - np.amin(im)) / (np.amax(im) - np.amin(im)) + 1e-4) * 255
        im = im.astype(np.uint8)
        images.append(im)
        #cv2.imwrite(config.dump_path + prefix + '_'  + str(j) + suffix + '.png', im)
    for j in range(5):
        images.append(np.zeros((const.frame_height, const.frame_width), dtype=np.uint8))
    imageio.mimsave(config.dump_path + prefix + '_' + str(label) + suffix + '.gif', images,duration=0.5)

def vis_frames(images,label,prefix,suffix):
    for j in range(5):
        images.append(np.zeros((const.frame_height, const.frame_width), dtype=np.uint8))
    imageio.mimsave(config.dump_path + prefix + '_' + str(label) + suffix + '.gif', images, duration=0.5)

if __name__ == '__main__':
    args = dict()
    args[data_args.gen_nearby_frame] = False;
    args[data_args.data_augmentation_enabled] = False
    vdz_dataset = UCFTupleLoader(args);
    words, contexts, labels = vdz_dataset.next(const.Subset.TRAIN, supervised=True)
    print(np.unique(np.argmax(labels,axis=1)))
    #print(labels)
    for batch_idx in range(contexts.shape[0]):
        vis_img(words[batch_idx,:],np.argmax(labels[batch_idx]),'p_'+str(batch_idx),'_img')
        vis_sod(contexts[batch_idx,:],np.argmax(labels[batch_idx]),'p_'+str(batch_idx),'_sod')
    sys.exit(1)
    # import time
    # for seed in range(1):
    #     np.random.seed(seed)
    #     print('Current using seed ',seed)
    #
    #     for i in range(10):
    #         start_time = time.time()
    #         words, contexts, labels = vdz_dataset.next(const.Subset.TRAIN, supervised=True)
    #         elapsed_time = time.time() - start_time
    #         print('elapsed_time :', elapsed_time)
    # words, contexts, labels = vdz_dataset.next(const.Subset.TRAIN, supervised=True)
    # for batch_idx in range(words.shape[0]):
    #     vis_img(words[batch_idx,:],np.argmax(labels[batch_idx]),'p_'+str(batch_idx),'_img')
    #     vis_sod(contexts[batch_idx,:],np.argmax(labels[batch_idx]),'p_'+str(batch_idx),'_sod')
    # sys.exit(1)
    #
    #
    # start_time = time.time()
    # words, contexts, lbls = vdz_dataset.next(const.Subset.VAL, fix_label=None, supervised=True)
    # elapsed_time = time.time() - start_time
    # print('elapsed_time :', elapsed_time)
    # # Some visualization for debugging purpose
    # #save_imgs('tuple_', words, lbls, '_img');
    # # save_pkls('tuple_', contexts, lbls, '_pkl');
    # print('Done')
