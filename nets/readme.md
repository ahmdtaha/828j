# Nets

This directory holds the networks investigated during the project. The Two-Stream Network uses AlexNet Tensorflow model + weights from [here](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/). The two stream network has two branches: word, context as shown in the figure below. The word branch takes RGB frame while the context branch takes encoded motion. Motion can be encoded as  stack of difference, [dynamic images](https://github.com/ahmdtaha/py_dynamic_imgs/) or optical flow

![](https://github.com/ahmdtaha/828j/blob/master/docs/imgs/two\_stream\_net.jpg)
 
The two branches,in the two-stream net, are named word\_, cntxt\_ so this is not siamese network. Yet, the network supports siamese architecture simply by fixing the same prefix across branches. The script has two mutually exclusion fully connected networks (i.e. two heads), for unsupervised and supervised training. A  boolean parameter control which "head" to be trained. Another boolean parameter controls whether to train the AlexNet CNN layers. This can be useful after unsupervised learning to fix pre-trained AlexNet weights.
   
![](https://github.com/ahmdtaha/828j/blob/master/docs/imgs/two\_heads.jpg)