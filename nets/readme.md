# Nets

This directoty holds the networks investigated during the project. The Two-Stream Network uses AlexNet Tensorflow model + weights from [here](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/). The two stream network has two branches: word, context as shown in the figure below. The word branch takes RGB Frame while the context branch motion encodes motion. Motion can be encoded in terms of stack of difference, dynamic images or optical flow

![](https://github.com/ahmdtaha/828j/blob/master/docs/imgs/two\_stream\_net.jpg)
 
In the two stream network, the two branches are named word\_, cntxt\_ so this is not siamese network. Yet, the network supports siamese architecture simply by fixing the same prefix across branches. The script has two mutually exclusion fully connected networks (i.e. two heads), for unsupervisied and supervisied training. A Constructor boolean parameter control which "head" to be trained. Another boolean parameter controls whether to train the AlexNet CNN layers. This can be useful during supervised learning to fix pre-trained AlexNet weights.
   
![](https://github.com/ahmdtaha/828j/blob/master/docs/imgs/two\_heads.jpg)