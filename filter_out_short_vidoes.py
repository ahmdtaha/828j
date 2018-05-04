import os.path as osp
import pickle

dataset_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucf101_downsampled'
# input_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/testlist01.txt'

# input_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/trainlist01-no_labels.txt'
# filtered_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/trainlist01_filtered.txt'

# input_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/trainlist02-no_labels.txt'
# filtered_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/trainlist02_filtered.txt'

# input_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/trainlist03-no_labels.txt'
# filtered_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/trainlist03_filtered.txt'

input_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/testlist01.txt'
filtered_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/testlist01_filtered.txt'

# input_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/testlist01.txt'
# filtered_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/testlist01_filtered.txt'

# input_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/testlist01.txt'
# filtered_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/testlist01_filtered.txt'

filtered_count = 0
none_count = 0
short_count = 0
total_count = 0
total_tuples = 0
fout = open(filtered_list_path, 'w')
for line in open(input_list_path, 'r'):
    total_count += 1
    video_name = line.strip()
    print('Processing file #%d: %s' % (total_count, video_name))
    filepath = osp.join(dataset_path, video_name[:-3] + 'pkl')
    with open(filepath, 'rb') as f:
        frames = pickle.load(f)
    if len(frames) >= 11:  # 6 frames with 5 intermediate center frames
        valid = True
        for frame in frames:
            if frame is None:
                valid = False
                break
        if valid:
            fout.write(video_name + '\n')
            total_tuples += (len(frames) - 1) // 2 - 4
        else:
            none_count += 1
    else:
        short_count += 1

fout.close()
filtered_count = short_count + none_count
print('Number of short videos = %d' % short_count)
print('Number of None videos = %d' % none_count)
print('Number of filtered out files = %d/%d' % (filtered_count, total_count))
print('Final number of tuples = %d' % total_tuples)

# =================================================================================================

# dataset_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucf101_downsampled'
# filtered_list_path = '/fs/vulcan-scratch/mmeshry/self_supervised_video_learning/datasets/ucfTrainTestlist/trainlist01_filtered.txt'
#
# # list subsampled videos with None entries => [UPDATE]: there were 4 videos, which I removed for now.
# for line in open(filtered_list_path, 'r'):
#     video_name = line.strip()
#     filepath = osp.join(dataset_path, video_name[:-3] + 'pkl')
#     with open(filepath, 'rb') as f:
#         frames = pickle.load(f)
#     for frame in frames:
#         if frame is None:
#             print(video_name)
#             break
