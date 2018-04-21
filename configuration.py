
import os
import logging


epoch_size = 500000
dataset_name = 'HMDB'
db_split = 1
use_two_stream  = True

dropout_rate = 0.2;
unsup_weight = 100;

dataset_dir = dataset_name
if(dataset_name == 'UCF50'):
    num_classes = 50
elif dataset_name == 'UCF101':
    num_classes = 101
    db_path = '/Users/ahmedtaha/Documents/dataset/UCF101_sampled'
    db_tuple_loader = 'data_sampling.ucf_tuple_loader.UCFTupleLoader'
elif dataset_name == 'HMDB':
    num_classes = 51
    db_path = '/Users/ahmedtaha/Documents/dataset/hmdb51_sampled'
    db_tuple_loader = 'data_sampling.hmdb_tuple_loader.HMDBTupleLoader'
elif dataset_name == 'honda':
    num_classes = 11
    db_path = '/Users/ahmedtaha/Documents/dataset/honda_100h'
    db_tuple_loader = 'data_sampling.honda_tuple_loader.HondaTupleLoader'

#unsupervised_num_classes = 4
unsupervised_num_classes = 7

tensorbaord_dir = './tb/'

def touch_dir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)


model_save_path = './model/cvprws/tmp'
touch_dir(model_save_path)
model_weights_filepath = '/Users/ahmedtaha/Documents/Models/bvlc_alexnet.npy'
model_save_name = "patch_model.ckpt"
tensorbaord_file = os.path.basename(os.path.normpath(model_save_path)) #'20180304-180936'


root_logger = logging.getLogger('root_logger')
root_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(model_save_path ,'debug.txt'),mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
root_logger.addHandler(fh)

dataset_path = ['/Users/ahmedtaha/Documents/dataset']
dump_path = '/Users/ahmedtaha/Documents/dataset/dump/'

o3n_weights_path = '/Users/ahmedtaha/Documents/dataset/weights/split3.mat'


## old ununsed flags
reduce_overfit = False