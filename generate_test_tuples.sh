#!/bin/bash

# ROOT_DIR="."
ROOT_DIR="/fs/vulcan-scratch/mmeshry/self_supervised_video_learning"

## The following is for the UCF101 dataset
DATASET="ucf101"
SPLIT_METADATA_DIR="$DATASET_PATH/ucfTrainTestlist"  # for UCF101

## The following is for the HMDB51 dataset
# DATASET="hmdb51"
# SPLIT_METADATA_DIR="$DATASET_PATH/testTrainMulti_7030_splits"  # for HMDB51

DATASET_PATH="$ROOT_DIR/datasets/$DATASET"
TEST_TUPLES_DIR="$DATASET_PATH/test_tuples"
NUM_THREADS=16

python generate_test_tuples.py \
    --dataset_name=$DATASET \
    --dataset_path=$DATASET_PATH \
    --split_metadata_dir=$SPLIT_METADATA_DIR  \
    --test_tuples_basedir=$TEST_TUPLES_DIR \
    --num_threads=$NUM_THREADS
