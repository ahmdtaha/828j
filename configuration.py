#texture_path = '/Users/ahmedtaha/UMD_Drive/Teaching/426-17/Projects/project4/data'
#data_path = '/Users/ahmedtaha/Documents/dataset/hmdb51_small'
#data_path = '/Users/ahmedtaha/Documents/dataset/UCF50_tuples_class'

import utils
epoch_size = 500000
dataset_name = 'honda'

dataset_dir = dataset_name
if(dataset_name == 'UCF50'):
    num_classes = 50
elif dataset_name == 'UCF101':
    num_classes = 101
elif dataset_name == 'HMDB':
    num_classes = 51
elif dataset_name == 'honda':
    num_classes = 11
unsupervised_num_classes = 4

tensorbaord_dir = './tb/'

#honda_session_path = '/mnt/work/honda_100h/'
honda_session_path = '/Users/ahmedtaha/Documents/dataset/honda_100h'
model_save_path = './model/unsup_ucf101_multi'
model_weights_filepath = '/Users/ahmedtaha/Documents/Models/bvlc_alexnet.npy'
model_save_name = "patch_model.ckpt"
tensorbaord_file = utils.get_last_part(model_save_path) #'20180304-180936'
dataset_path = ['/Users/ahmedtaha/Documents/dataset']
dump_path = '/Users/ahmedtaha/Documents/dataset/dump/'

o3n_weights_path = '/Users/ahmedtaha/Documents/dataset/weights/split3.mat'


