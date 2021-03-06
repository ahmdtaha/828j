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
from utils.logger import root_logger as logger

val_dir = 'val'


class HondaTupleLoader:

    def load_subset(self,subset):
        with open(config.db_path+'/'+subset+'_session.txt', 'r') as fin:
            sessions = fin.read().strip().split('\n')
        num_sessions = len(sessions)
        sessions_annotations = [None] * num_sessions

        for vdz in range(num_sessions):
            vdz_annotation_path = os.path.join(config.db_path,'labels/'+sessions[vdz]+'_goal.pkl')
            sessions_annotations[vdz] = os_utils.pkl_read(vdz_annotation_path)

        return sessions,num_sessions,sessions_annotations
    def __init__(self, args):
        self.train_sessions,self.num_train_sessions,self.train_sessions_annotations = self.load_subset( 'train');
        self.val_sessions, self.num_val_sessions, self.val_sessions_annotations = self.load_subset(val_dir);

        print('Train ',self.num_train_sessions, ' Val ',self.num_val_sessions)

        logger.info('Train ' + str(self.num_train_sessions) + ' Val ' + str(self.num_val_sessions))

    def imgs2sod(self,imgs,ordered=True):

        rand_crop = np.random.rand();
        y = int(rand_crop * (imgs[0].shape[0] - imgs[0].shape[0] * 0.8))
        x = int(rand_crop * (imgs[0].shape[1] - imgs[0].shape[1] * 0.8))

        rand_rgb_channel = np.random.choice(3);
        stack_diff = np.zeros((const.frame_height, const.frame_width, const.context_channels))

        ## Random order
        # if (ordered):
        #     c = -1;
        #     frames_order = np.arange(const.context_channels + 1)
        # else:
        #     c = 0;
        #     frames_order = np.random.permutation(const.context_channels + 1)

        # Swtich two frames only to make it more difficult
        frames_order = np.arange(const.context_channels + 1)
        c = -1;
        if (not ordered):
            c = np.random.choice(len(frames_order), 1, replace=False);
            n = (c + 1) % len(frames_order)
            frames_order[c], frames_order[n] = frames_order[n], frames_order[c]

        for i in range(const.context_channels):
            current_frame = imgs[frames_order[i]][y:y+int(imgs[0].shape[0] * 0.8), x:x+int(imgs[0].shape[1] * 0.8),rand_rgb_channel];
            next_frame = imgs[frames_order[i+1]][y:y+int(imgs[0].shape[0] * 0.8), x:x+int(imgs[0].shape[1] * 0.8), rand_rgb_channel];
            current_frame = cv2.resize(current_frame, (const.frame_height, const.frame_width))
            next_frame = cv2.resize(next_frame, (const.frame_height, const.frame_width))

            stack_diff[:,:,i] = current_frame.astype(np.int32) - next_frame.astype(np.int32);

        return stack_diff;

    def read_frame(self,idx,session):
        img_path = os.path.join(config.db_path, 'frames', session, 'frame_%04d.jpg' % (idx));
        # print((center_idx+idx)," ",end="")
        if (not os.path.exists(img_path)):
            img_path = os.path.join(config.db_path, 'frames', session,
                                    'frame_%05d.jpg' % (idx));

        try:
            img = imageio.imread(img_path);
        except:
            img = np.zeros((const.frame_height, const.frame_width,const.frame_channels))
        return img

    def get_context(self,center_idx,session,ordered=True,n=1):
        imgs = []
        # print('Context ')
        for idx in [-5,-3,-1,1,3,5]:
            img = self.read_frame(center_idx + idx,session)
            imgs.append(img)
        # print("")
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
        img = self.read_frame(frame_idx, session)
        return self.augment_img(img)

    def get_valid_event_balanced(self,pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations):

        session_idx = pos_tuple[batch_idx];
        session_annotations = current_sessions_annotations[session_idx]
        num_annotations = len(session_annotations['s'])
        current_session_possible_events = np.unique(session_annotations['G']);
        sessions_possible_events = config.num_classes
        event_type = int(annotation_idx[batch_idx] * sessions_possible_events)
        while(event_type  not in current_session_possible_events):
            event_type = (event_type  +1) % sessions_possible_events;

        balanced_events =[idx for idx,evnt in enumerate(session_annotations['G']) if evnt == event_type]
        event_idx = np.random.choice(balanced_events)
        #event_idx = int(annotation_idx[batch_idx] * num_annotations)
        if(event_idx >= num_annotations-1):
            event_idx = num_annotations-2
        invalid_event = True
        while invalid_event:
            try:
                event_start = session_annotations['s'][event_idx]
                event_end = session_annotations['s'][event_idx + 1]
                event_goal = session_annotations['G'][event_idx]
                center_idx = (event_start + event_end) // 2
                if event_end - event_start > 5 and center_idx > 5 and center_idx < session_annotations['s'][-1]-6:  # ignore short (background) clips
                    invalid_event = False;  ## found a valid event

                    return event_start,event_end,event_goal;
                else:
                    invalid_event = True;  ## Keep looking for valid event
                    event_idx = (event_idx - 1) % num_annotations
            except:
                invalid_event = True;  ## Keep looking for valid event
                event_idx = (event_idx - 1) % num_annotations

        return None;

    def get_valid_event(self,pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations):
        return self.get_valid_event_balanced(pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations)

        session_idx = pos_tuple[batch_idx];
        session_annotations = current_sessions_annotations[session_idx]
        num_annotations = len(session_annotations['s'])
        event_idx = int(annotation_idx[batch_idx] * num_annotations)
        if(event_idx >= num_annotations-1):
            event_idx = num_annotations-2
        invalid_event = True
        while invalid_event:
            try:
                event_start = session_annotations['s'][event_idx]
                event_end = session_annotations['s'][event_idx + 1]
                event_goal = session_annotations['G'][event_idx]
                center_idx = (event_start + event_end) // 2
                if event_end - event_start > 5 and center_idx > 5 and center_idx < session_annotations['s'][-1]-6:  # ignore short (background) clips
                    invalid_event = False;  ## found a valid event

                    return event_start,event_end,event_goal;
                else:
                    invalid_event = True;  ## Keep looking for valid event
                    event_idx = (event_idx - 1) % num_annotations
            except:
                invalid_event = True;  ## Keep looking for valid event
                event_idx = (event_idx - 1) % num_annotations

        return None;
    def sample_valid_center_frame(self,event_start,event_end):
        trim = 5;
        if(event_end - event_start <= (2* trim+1)):
            center = (event_start + event_end) // 2
            #print('Center ', center, ' from ', event_start, ' end', event_end)
            return center
        #print(' from ', event_start, ' end', event_end)
        rand_center = np.random.choice(range(trim+1,event_end - event_start-trim));
        center = (event_start + event_end) //2
        rand_center += event_start
        #print('Center ',center, ' while rand center ',rand_center, ' from ',event_start, ' end',event_end)
        return rand_center

    def load_pos_neg_tuple(self, pos_tuple,neg_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations,
                       ordered=True):
        pos_session = current_sessions[pos_tuple[batch_idx]]
        neg_session = current_sessions[neg_tuple[batch_idx]]
        pos_event_start, pos_event_end, pos_event_goal = self.get_valid_event(pos_tuple, annotation_idx, batch_idx,
                                                                              current_sessions,
                                                                              current_sessions_annotations)

        neg_event_start, neg_event_end, neg_event_goal = self.get_valid_event(neg_tuple, annotation_idx, batch_idx,
                                                                              current_sessions,
                                                                              current_sessions_annotations)

        pos_center_idx = self.sample_valid_center_frame(pos_event_start , pos_event_end)
        neg_center_idx = self.sample_valid_center_frame(neg_event_start  , neg_event_end) #(neg_event_start + neg_event_end) // 2

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
                img_path = os.path.join(config.db_path, 'frames', neg_session, 'frame_%04d.jpg' % (neg_center_idx+i));
                img = imageio.imread(img_path);
                imgs.append(img)

            vis_frames(imgs,label, 'p_' + str(batch_idx), '_imgs')
        return word,context




    def load_pos_tuple(self, pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations,ordered=True,n=1):
        session = current_sessions[pos_tuple[batch_idx]]
        event_start, event_end, event_goal = self.get_valid_event(pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations)
        center_idx = self.sample_valid_center_frame(event_start , event_end) #(event_start + event_end) //2
        center_idxs = [center_idx]*n
        half_n = n//2
        for i in range(half_n):
            center_idxs[i] = center_idx - 11 * (half_n-i)
            center_idxs[-i-1] = center_idx + 11 * (half_n-i)
        # print(center_idxs)
        word = np.zeros((const.frame_height, const.frame_width, const.frame_channels * n))
        context = np.zeros((const.frame_height, const.frame_width, const.context_channels * n))

        for i in range(n):
            word[:,:,i*const.frame_channels :(i+1)*const.frame_channels ] = self.get_img(center_idxs[i], session)
            context[:,:,i*const.context_channels:(i+1)*const.context_channels] = self.get_context(center_idxs[i], session, ordered)

        return word,context,event_goal;


    def unsupervised_next(self,subset):

        subset_size = 0
        if (subset == const.Subset.TRAIN):
            subset_size = self.num_train_sessions
            current_sessions = self.train_sessions
            current_sessions_annotations = self.train_sessions_annotations
        elif (subset == const.Subset.VAL):
            subset_size = self.num_val_sessions
            current_sessions = self.val_sessions
            current_sessions_annotations = self.val_sessions_annotations
        elif (subset == const.Subset.TEST):
            img_set = self._test_activities

        pos_tuple = np.random.randint(low=0, high=subset_size, size=(const.batch_size));
        annotation_idx = np.random.rand(const.batch_size);

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
                word,context,_ = self.load_pos_tuple(pos_tuple, annotation_idx, batch_idx, current_sessions,
                                                     current_sessions_annotations,ordered=True);
            elif (sampling_ratio[batch_idx] < 0.5):
                ## Same Image , UnOrdered
                labels[batch_idx] = 1;
                word, context,_ = self.load_pos_tuple(pos_tuple, annotation_idx, batch_idx, current_sessions,
                                                    current_sessions_annotations,ordered=False);
            elif (sampling_ratio[batch_idx] < 0.75):
                ## Different Image , Ordered
                labels[batch_idx] = 2;
                word, context = self.load_pos_neg_tuple(pos_tuple,neg_tuple, annotation_idx, batch_idx, current_sessions,
                                                    current_sessions_annotations, ordered=True);
            else:
                ## Different Image , UnOrdered
                labels[batch_idx] = 3;
                word, context = self.load_pos_neg_tuple(pos_tuple, neg_tuple, annotation_idx, batch_idx,current_sessions,
                                                        current_sessions_annotations, ordered=False);

            sample_count[labels[batch_idx]] +=1
            words[batch_idx, :, :] = word
            contexts[batch_idx, :, :] = context

        #print(sample_count)
        labels_hot_vector = os_utils.hot_one_vector(labels,4);
        return words, contexts, labels_hot_vector
    # def dump(self, pos_tuple, annotation_idx, batch_idx, current_sessions, current_sessions_annotations,
    #                       ordered, queue):
    #     queue.put(True)
    #     queue.put(np.zeros((const.frame_height,1)))
    #     #queue.put(np.zeros((const.frame_height, const.frame_height, 5)))
    #     #queue.put(6)

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
        subset_size = 0
        if (subset == const.Subset.TRAIN):
            subset_size = self.num_train_sessions
            current_sessions = self.train_sessions
            current_sessions_annotations = self.train_sessions_annotations
        elif (subset == const.Subset.VAL):
            subset_size = self.num_val_sessions
            current_sessions = self.val_sessions
            current_sessions_annotations = self.val_sessions_annotations
        elif (subset == const.Subset.TEST):
            img_set = self._test_activities

        pos_tuple = np.random.randint(low=0, high=subset_size, size=(const.batch_size));
        annotation_idx = np.random.rand(const.batch_size);



        labels = np.zeros((const.batch_size), dtype=np.int32)

        sample_count = np.zeros(config.num_classes);

        #queue_size = const.batch_size;
        # queues = [None] * queue_size
        # processes = [None] * queue_size
        # print('queue_size ',queue_size)

        # queue_size = 5
        # pool = Pool(processes=queue_size )
        # res = [None] * queue_size
        n = config.sod_span;
        words = np.zeros((const.batch_size, const.frame_height, const.frame_width, const.frame_channels * n))
        contexts = np.zeros((const.batch_size,const.frame_height, const.frame_width, const.context_channels * n))
        for batch_idx in np.arange(0,const.batch_size):


            word, context, goal = self.load_pos_tuple(pos_tuple, annotation_idx, batch_idx, current_sessions,
                                                      current_sessions_annotations, ordered=True,n=n);
            labels[batch_idx] = goal;
            words[batch_idx, :, :,:] = word
            contexts[batch_idx, :, :,:] = context

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
        #print(np.histogram(labels))
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
    for j in range(stack_diff.shape[2]):
        im = stack_diff[:, :, j];
        im = ((im - np.amin(im)) / np.amax(im) - np.amin(im)) * 255
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
    vdz_dataset = HondaTupleLoader(args);
    import time
    for seed in range(100):
        np.random.seed(seed)
        print('Current using seed ',seed)

        for i in range(1):
            start_time = time.time()
            words, contexts, labels = vdz_dataset.next(const.Subset.TRAIN, supervised=True)
            elapsed_time = time.time() - start_time
            print('elapsed_time :', elapsed_time)

    #words, contexts, labels = vdz_dataset.next(const.Subset.TRAIN, supervised=True)
    # for batch_idx in range(words.shape[0]):
    #     vis_img(words[batch_idx,:,:,config.sod_span//2:config.sod_span//2+3],np.argmax(labels[batch_idx]),'p_'+str(batch_idx),'_img')
    #     vis_sod(words[batch_idx, :], np.argmax(labels[batch_idx]), 'i_' + str(batch_idx), '_frms')
    #     vis_sod(contexts[batch_idx,:],np.argmax(labels[batch_idx]),'i_'+str(batch_idx),'_sod')
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
