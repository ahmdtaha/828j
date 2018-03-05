#texture_path = '/Users/ahmedtaha/UMD_Drive/Teaching/426-17/Projects/project4/data'
#data_path = '/Users/ahmedtaha/Documents/dataset/hmdb51_small'
#data_path = '/Users/ahmedtaha/Documents/dataset/UCF50_tuples_class'

supervised = True
dataset_name = 'UCF50'
if(supervised):
    dataset_dir = dataset_name
    num_classes = 50
else:
    dataset_dir = dataset_name
    num_classes = 4

tensorbaord_dir = './tb/'


model_save_path = './model/o3n'
model_weights_filepath = '/Users/ahmedtaha/Documents/Models/bvlc_alexnet.npy'
model_save_name = "patch_model.ckpt"
tensorbaord_file = '20180304-180936'
dataset_path = ['/Users/ahmedtaha/Documents/dataset']
dump_path = '/Users/ahmedtaha/Documents/dataset/dump/'

o3n_weights_path = '/Users/ahmedtaha/Documents/dataset/weights/split3.mat'


