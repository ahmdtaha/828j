import sys
sys.path.append('../')
import utils
import os
import numpy as np

if __name__ == '__main__':
    path = utils.get_dataset_path('UCF50')
    dirs = utils.get_dirs(path);
    train_list = []
    val_list = []
    test_list = []
    train_set_percentage = 0.7
    val_set_percentage_from_training = 0.2
    for d in dirs:
        current_dir = os.path.join(path, d);
        files = utils.get_files(current_dir, '.avi', append_base=False);
        files = [os.path.join(d,f) for f in files];
        np.random.shuffle(files)

        num_train = int(len(files) * train_set_percentage);
        num_val = int(num_train * val_set_percentage_from_training)
        train_list.extend(files[0:num_train-num_val]) ;
        val_list.extend(files[num_train-num_val:num_train]);
        test_list.extend(files[num_train:]);
        print(len(files),len(train_list),len(val_list ),len(test_list))

    save_path = path+'_lists'
    utils.touch_dir(save_path)
    utils.txt_write(save_path+'/trainlist.txt',train_list);
    utils.txt_write(save_path + '/vallist.txt', val_list);
    utils.txt_write(save_path + '/testlist.txt', test_list);


