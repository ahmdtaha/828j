
import os


epoch_size = 500000
dataset_name = 'honda'
db_split = 1
use_two_stream  = False

dropout_rate = 0.5;
unsup_weight = 100;

dataset_dir = dataset_name

sod_span = 5;
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

unsupervised_num_classes = 4
#unsupervised_num_classes = 7

tensorbaord_dir = './tb/'

def touch_dir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)


model_save_path = './model/comb_unsup_ucf101_sf_dp05'
touch_dir(model_save_path)
model_weights_filepath = '/Users/ahmedtaha/Documents/Models/bvlc_alexnet.npy'
model_save_name = "patch_model.ckpt"
tensorbaord_file = os.path.basename(os.path.normpath(model_save_path)) #'20180304-180936'



dataset_path = ['/Users/ahmedtaha/Documents/dataset']
dump_path = '/Users/ahmedtaha/Documents/dataset/dump/'

o3n_weights_path = '/Users/ahmedtaha/Documents/dataset/weights/split3.mat'

checkpoint_frequency = 2

## old ununsed flags
reduce_overfit = False