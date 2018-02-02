# 828J Project Code

This repository maintains the code for 828j project. It is divided into two parts: data sampling, and training/nets

./data_sampling directory contains code for data sampling where high motion aware tuples are sampled. This should contain also code for spatial jittering and channel splitting. More details [here](https://github.com/ahmdtaha/828j/blob/master/data_sampling/readme.md)

./net directory contains a Two-Stream network built upon AlexNet. More details [here](https://github.com/ahmdtaha/828j/blob/master/nets/readme.md)

unsupervised\_train.py and supervised\_train.py use the two-stream network and ./data_sampling/tuple_loader to train the network during unsupervised (aka self-supervised) and supervised mode respectively.


