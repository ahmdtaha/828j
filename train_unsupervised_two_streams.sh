base_dir_write='/fs/vulcan-scratch/ahmdtaha/self_supervised_video_learning'
base_dir_read='/fs/vulcan-scratch/mmeshry/self_supervised_video_learning'


#dataset_path="$base_dir_read/datasets/hmdb51_downsampled"
dataset_path="$base_dir_read/datasets/ucf101_downsampled"

split1_train_file="$base_dir_read/datasets/ucfTrainTestlist/trainlist01_filtered.txt"
split1_val_file="$base_dir_read/datasets/ucfTrainTestlist/testlist01_filtered.txt"

split2_train_file="$base_dir_read/datasets/ucfTrainTestlist/trainlist02_filtered.txt"
split2_val_file="$base_dir_read/datasets/ucfTrainTestlist/testlist02_filtered.txt"

split3_train_file="$base_dir_read/datasets/ucfTrainTestlist/trainlist03_filtered.txt"
split3_val_file="$base_dir_read/datasets/ucfTrainTestlist/testlist03_filtered.txt"
activities_path="$base_dir_read/datasets/ucfTrainTestlist/activities"

input_train_list_filepaths=$split1_train_file
input_val_list_filepaths=$split1_val_file
# input_list_filepaths="$split1_train_file $split1_train_file $split3_train_file"

supervision_mode="unsupervised"
# run_mode="train"

experiment_name='unsup_ucf101_multi'
model_save_path="$base_dir_write/model/$experiment_name"
tensorboard_save_path="$base_dir_write/tb/$experiment_name"

python train_two_streams.py \
    --dataset_path $dataset_path \
    --input_train_list_files $input_train_list_filepaths \
    --input_val_list_files $input_val_list_filepaths \
    --activities_file_path $activities_path \
    --model_save_path $model_save_path \
    --tensorboard_save_path $tensorboard_save_path \
    --supervision_mode $supervision_mode
