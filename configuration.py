
import utils.os_utils as os_utils
epoch_size = 500000
dataset_name = 'UCF101'


dataset_dir = dataset_name
if(dataset_name == 'UCF50'):
    num_classes = 50
elif dataset_name == 'UCF101':
    num_classes = 101
    db_path = '/Users/ahmedtaha/Documents/dataset/UCF101_sampled'
elif dataset_name == 'HMDB':
    num_classes = 51
    db_path = '/Users/ahmedtaha/Documents/dataset/hmdb51_sampled'
elif dataset_name == 'honda':
    num_classes = 11
    db_path = '/Users/ahmedtaha/Documents/dataset/honda_100h'
unsupervised_num_classes = 4

tensorbaord_dir = './tb/'


model_save_path = './model/unsup_sup_hmdb'
model_weights_filepath = '/Users/ahmedtaha/Documents/Models/bvlc_alexnet.npy'
model_save_name = "patch_model.ckpt"
tensorbaord_file = os_utils.get_last_part(model_save_path) #'20180304-180936'
dataset_path = ['/Users/ahmedtaha/Documents/dataset']
dump_path = '/Users/ahmedtaha/Documents/dataset/dump/'

o3n_weights_path = '/Users/ahmedtaha/Documents/dataset/weights/split3.mat'


