#!/bin/bash

# ROOT_DIR="."
ROOT_DIR="/fs/vulcan-scratch/mmeshry/self_supervised_video_learning"

## The following is for the UCF101 dataset
DATASET="ucf101"
SPLIT_METADATA_DIR="$ROOT_DIR/datasets/ucfTrainTestlist"  # for UCF101

## The following is for the HMDB51 dataset
# DATASET="hmdb51"
# SPLIT_METADATA_DIR="$ROOT_DIR/datasets/testTrainMulti_7030_splits"  # for HMDB51

DATASET_PATH="$ROOT_DIR/datasets/$DATASET"
TEST_TUPLES_DIR="$DATASET_PATH/test_tuples_py36_lm"
# TRAINED_MODEL_PATH="$ROOT_DIR/models/$DATASET.cpkt"
# TRAINED_MODEL_PATH=/vulcan/scratch/ahmdtaha/model/supervised_ucf101_20hrs/patch_model.ckpt
# TRAINED_MODEL_PATH="$ROOT_DIR/model/3splits_noAug_4epochs/patch_model.ckpt"
TRAINED_MODEL_PATH="$ROOT_DIR/model/split1_aug_10.5epochs/patch_model.ckpt"
# TRAINED_MODEL_PATH="$ROOT_DIR/model/sup_archive/taha_code_rerun_4epochs/patch_model.ckpt"
# TRAINED_MODEL_PATH=/vulcan/scratch/ahmdtaha/model/unsupervised_supervised_ucf101/patch_model.ckpt
# TRAINED_MODEL_PATH=/vulcan/scratch/ahmdtaha/model/unsupervised_ucf101/patch_model.ckpt

python test_supervised.py \
    --dataset_name=$DATASET \
    --dataset_path=$DATASET_PATH \
    --split_metadata_dir=$SPLIT_METADATA_DIR  \
    --test_tuples_basedir=$TEST_TUPLES_DIR \
    --model_save_dir=$TRAINED_MODEL_PATH
