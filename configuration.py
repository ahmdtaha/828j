#texture_path = '/Users/ahmedtaha/UMD_Drive/Teaching/426-17/Projects/project4/data'
#data_path = '/Users/ahmedtaha/Documents/dataset/hmdb51_small'
#data_path = '/Users/ahmedtaha/Documents/dataset/UCF50_tuples_class'

import utils
epoch_size = 500000
# dataset_name = 'UCF50'
dataset_name = 'UCF101'

dataset_dir = dataset_name
if(dataset_name == 'UCF50'):
    num_classes = 50
elif dataset_name == 'UCF101':
    num_classes = 101
elif dataset_name == 'HMDB':
    num_classes = 51
unsupervised_num_classes = 4

# tensorbaord_dir = './tb/'
# 
# 
# model_save_path = './model/unsup_ucf101_multi'
# model_weights_filepath = '/Users/ahmedtaha/Documents/Models/bvlc_alexnet.npy'
# model_save_name = "patch_model.ckpt"
# tensorbaord_file = utils.get_last_part(model_save_path) #'20180304-180936'
# dataset_path = ['/Users/ahmedtaha/Documents/dataset']
# dump_path = '/Users/ahmedtaha/Documents/dataset/dump/'
# 
# o3n_weights_path = '/Users/ahmedtaha/Documents/dataset/weights/split3.mat'

basedir = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning'
# tensorbaord_dir = basedir + '/tb/'

# model_save_path = basedir + '/model/sup_ucf101_multi'
# model_save_path = basedir + '/model/unsup_ucf101_multi'

model_weights_filepath = basedir + '/data/bvlc_alexnet.npy'
model_save_name = "patch_model.ckpt"
# tensorbaord_file = utils.get_last_part(model_save_path) #'20180304-180936'
# dataset_path = [basedir + '/dataset']
dataset_path = ['/vulcan/scratch/ahmdtaha']
dump_path = basedir + '/dataset/dump/'

o3n_weights_path = '/Users/ahmedtaha/Documents/dataset/weights/split3.mat'
