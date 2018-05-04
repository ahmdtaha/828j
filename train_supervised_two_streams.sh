base_dir='/fs/vulcan-scratch/mmeshry/self_supervised_video_learning'
dataset_path="$base_dir/datasets/ucf101_downsampled"
split1_train_file="$k_base_dir/datasets/ucfTrainTestlist/trainlist01_filtered.txt"
split2_train_file="$k_base_dir/datasets/ucfTrainTestlist/trainlist02_filtered.txt"
split3_train_file="$k_base_dir/datasets/ucfTrainTestlist/trainlist03_filtered.txt"
input_train_list_filepaths=$split1_train_file
input_val_list_filepaths=$split1_train_file  # TODO: change to validation set
# input_list_filepaths="$split1_train_file $split1_train_file $split3_train_file"
activities_path="$k_base_dir/datasets/ucfTrainTestlist/activities"
supervision_mode="supervised"
# run_mode="train"

python train_two_streams.py \
    --dataset_path $dataset_path \
    --input_train_list_files $input_list_filepaths \
    --input_val_list_files $input_list_filepaths \
    --activities_file_path $activities_path \
    --supervision_mode $supervision_mode
                            
