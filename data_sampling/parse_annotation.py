import sys
sys.path.append('../')
import os
import glob
import pympi
import pickle
import h5py
import numpy as np
import pdb
import pandas as pd
import data_sampling.honda_labels as honda_lbls
import configuration as config
label_dir = config.honda_session_path+'/labels/'
feature_dir = config.honda_session_path+'/features/'

label_dict = {}

def convert_seg(seg):
    """
    Convert original segmentation vector

    Input
        seg    -   original segmentation vector with size N

    Output
        s  -  starting position of each segment, list with size m+1, m is the number of segment
        G  -  label of each segment, list with size m 
    """

    N = seg.shape[0]

    s = [0]
    G = [seg[0]]
    for i in range(1, N):
        if not seg[i] == seg[i-1]:
            s.append(i)
            G.append(seg[i])
    s.append(N)

    return s, G

def parse_annotation(layer, session_id, N):
    """
    N - number of frames of the video
    videos are down-sampled to 3fps

    Annotation is not so precise, +-3 seconds offset is possible
    """

#        session_folder = self.session_template.format(self.cfg.DATA_ROOT,
#                                                    session_id[:4],
#                                                    session_id[4:6],
#                                                    session_id[6:8],
#                                                    session_id)
#        annotation_filename = glob.glob(session_folder + 'annotation/event/*eaf')
#        if len(annotation_filename) == 0:
#            annotation_filename = glob.glob(session_folder + 'annotation/event/1/*eaf')
    annotation_path = config.honda_session_path+'EAF/event/{0}-{1}-{2}-{3}-{4}-*.eaf'.format(
            session_id[:4],
            session_id[4:6],
            session_id[6:8],
            session_id[8:10],
            session_id[10:12]);
    annotation_filename = glob.glob(annotation_path)
    annotation_filename = annotation_filename[0]

    label = np.zeros((N,), dtype='int32')
    eafob = pympi.Elan.Eaf(annotation_filename)
    # num_event_found = 0
    # found_events ={}
    for annotation in eafob.get_annotation_data_for_tier(layer):
        name = annotation[2].strip()
        #print(name)
        # manually fix some bug in annotation
        if name == '':
            continue
        ###### remove "parking" events ########
        if name.split(' ')[-1] == 'park':
            continue

        if not name in label_dict:
            label_dict[name] = len(label_dict.keys())
        
        start = int(np.round(annotation[0] / 1000.)) * 3

        end = int(np.round(annotation[1] / 1000.)) * 3 

        ###### remove short events ########
        if end - start < 5:
            continue


        if start>=0 and end<N:
            label[start:end+1] = label_dict[name]
            # print(start,'\t',end,'\t',name)
            # num_event_found+=1
            # found_events[start] = name;
        elif start<N and end>0:    # partially overlapped
            print("Partial adjustment: ", start, end, N)
            start = max(start, 0)
            end = min(N-1, end)
            label[start:end+1] = label_dict[name]
        else:
            print("Skip this: ", start, end, N)
            #raise ValueError("Length error!")
    # print('num_event_found ',num_event_found)
    #print(label)
    # keys = sorted(found_events.keys())
    # for key in keys:
    #     print(key,'\t',found_events[key])

    return label


"""
extract unoverlapped events, we are intereseted in the layers:
u'\u88ab\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Stimuli-driven'
u'\u4e3b\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Goal-oriented'
u'\u539f\u56e0 Cause'
"""


#layers = [u'\u88ab\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Stimuli-driven',
layers = u'\u4e3b\u52d5\u7684\u99d5\u99db\u884c\u70ba Operation_Goal-oriented'

#label_dict['background'] = 0
label_dict = honda_lbls.honda_labels2num

feature_files = glob.glob(feature_dir+'*sensors.npy')
feature_files = sorted(feature_files)

for fin in feature_files:
    session_id = os.path.basename(fin).split('_')[0]
    print("Session: " + session_id)

    N = np.load(fin).shape[0]

    label = parse_annotation(layers, session_id, N)

    ################# Note: remove some rare events #############
#    label[0,label[0,:]==6] = 0    # avoid parked car
#    label[0,label[0,:]==7] = 0    # avoid bycyclist
#    label[1,label[1,:]==12] = 0    # park


    s, G = convert_seg(label)

    pickle.dump({'label': label, 's':s, 'G':G}, open(label_dir+session_id+'_goal.pkl', 'wb'))


# print ("Save label dictionary")

# num2label={}
# for key in label_dict:
#     num2label[label_dict[key]] = key
# pickle.dump({'num2label':num2label, 'label2num':label_dict},
#         open(label_dir+'label_goal.pkl', 'wb'))


