# Smoothed Odd One Out

Odd One Out network predicts the odd clip among n clips. The odd clip is a wrong clip in which frames are unordered. The smooth\_odd\_out.py create such network using n=3 as default. In fact, smooth\_odd\_out.py has two inner braches. The first predicts the odd clip while the second promote smoothness between nearby clips.

The following figure shows full structure of the network. First **2*n** clips are fed to the network. Smooth_O3N block contains AlexNet five conv layers and a final embedding dense layer of dimension 128 as suggested by O3N paper. These embedding are fused using sum of difference, then fed to FCN to train the O3N loss (L1). The same embedding trains smooth loss (L2) that promote close embedding between nearby clips
![](https://github.com/ahmdtaha/828j/blob/master/docs/imgs/O3N/full\_structure.jpg)

The following figure shows the Smooth_O3N block in details. In this image, the network takes two clips for odd one out and two clips to impose smoothness-- total four clips. Notice the three branches coming out of every conv layer, these are the conv weights learned. Thus, this is a siamese network -- convolution weights are shared between all four branches.
![](https://github.com/ahmdtaha/828j/blob/master/docs/imgs/O3N/o3n_conv_layer.jpg)
 