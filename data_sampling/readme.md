# Data Sampling

This package pre-processes the data needed during training the networks. The are three main pre-processing steps (1) motion aware sampling, (2) spatial jittering, (3) channel splitting.

Despite being trivial spatial jittering, channel splitting are not implemented yet. At the current stage, architecture investigation is the main task. So fine-tuning steps are pushed for later project stage. For motion aware sampling requires optical flow computation from [pyflow, a python lib,](https://github.com/pathak22/pyflow) is used.

To prepare data, the following scripts needs to be executed


### dataset_divider.py
This script generates training, validation and testing lists for a given dataset. The dataset path and the partitioning percentage variables are defined inside the main function. Generated sample [here](https://www.dropbox.com/s/4nd61p0vjbgmvet/UCF50_lists.zip?dl=0)

### motion_aware.py
This script calculates the optical flow magnitude for each video frame to sense high motion. The magnitude is normalized, across all frames, to be used as a sampling probability. A clip is a group of frames used to encodes motion. 
A clip extends throughout 60 frames, i.e. 2 seconds, following [Misra et al.](https://arxiv.org/pdf/1603.08561.pdf) guidelines t_max = 60. The generated flow magnitude is saved in a pickle format for future usage. This script processes multiple videos, by default 25 video, in parallel to save time. Generated sample [here](https://www.dropbox.com/s/ehguv70r0cwlgvw/UCF50_pkls.zip?dl=0). 

### tuple_generator.py
This script generates tuples, of RGB frames and a clip stack of difference, to be used training. It takes an optional parameter ,--is_training, to determine whether to generate training or validation tuples. Generated sample [here](https://www.dropbox.com/s/ty0i3b9qsj989ig/UCF50_tuples_class.zip?dl=0). 

### tuple_loader.py
This script is used during training to create a mini-batch of tuples with labels. 






